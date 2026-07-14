import heapq
import json
from pathlib import Path

import numpy as np

# ========== 参数 ==========
NUM_MATH_PROBLEMS = 100      # math 类问题个数
NUM_QA_PROBLEMS = 100        # qa 类问题个数
PARALLEL = 32               # 同时推理的 group 数（rollout batch size）
NUM_BATCHES = 300           # 模拟的 batch 数量
N_SAMPLES_PER_PROMPT = 8    # 每个 group 内的 sample 数（n_samples_per_prompt）
TRAIN_TIME_PER_BATCH = 160.0  # buffered 模拟中，每个 train step 占用的 wall-clock 时间
TRAIN_TIME_AFTER_BATCH = None  # 设置为 batch index 后，可模拟训练耗时阶段性变化
TRAIN_TIME_AFTER_VALUE = None  # TRAIN_TIME_AFTER_BATCH 之后使用的 train time
MATH_MEAN_TIME = 150.0      # 单个 math sample 推理平均耗时
QA_MEAN_TIME = 40.0         # 单个 qa sample 推理平均耗时
MATH_STD_TIME = 60.0        # 单个 math sample 推理耗时标准差
QA_STD_TIME = 40.0          # 单个 qa sample 推理耗时标准差；搜索任务用大 std 模拟长尾

# 对齐训练脚本里的 dynamic alternation 参数。
MATH_RATIO = 0.5           # 目标 math 比例，也对应 --math-ratio
DYNAMIC_ALPHA = 1         # lag-based ratio 权重；final=(1-alpha)*math_ratio + alpha*lag_ratio
MIN_MATH_RATIO = 0.2        # 对应 --dynamic-alternation-min-math-ratio
MAX_MATH_RATIO = 0.8        # 对应 --dynamic-alternation-max-math-ratio
WARMUP_STEPS = 5            # 对应 --dynamic-alternation-warmup-steps
BLOCK_ALTERNATION_SIZE = 16  # fixed block 模式下，连续 dispatch 多少个 group 后切换 task
MAX_CONSECUTIVE_DISPATCH = 16  # cap 模式下，同一 task 最多连续 dispatch 的 group 数
WINDOW_RATIO_SIZE = 32       # window 模式下，控制最近多少个 dispatch 的局部比例
LAG_PENALTY_BETA = 0.5       # debt_lag 模式下，lag penalty 相对比例欠账的权重

# 时间分布: "gamma", "lognormal", "exponential", "pareto", "uniform", "constant", "real"
DISTRIBUTION = "real"
REAL_PROFILE_DIR = Path(__file__).resolve().parent / "debug" / "profile"
REAL_GROUP_TIME_FILES = {
    "math": REAL_PROFILE_DIR / "profile_math_groups8_rollout0.jsonl",
    "qa": REAL_PROFILE_DIR.parent / "profile_trainqa_sglang32_mem05" / "profile_qa_groups_rollout0.jsonl",
}
REAL_GROUP_TIME_SCALE = {
    "math": 1.0,
    "qa": 2.0,
}
# pareto 参数 (shape alpha) — alpha 越小尾部越重
PARETO_ALPHA = 2.0


_REAL_GROUP_TIME_CACHE = None


def load_real_group_times():
    """Load measured group completion times from profile.sh output."""
    global _REAL_GROUP_TIME_CACHE
    if _REAL_GROUP_TIME_CACHE is not None:
        return _REAL_GROUP_TIME_CACHE

    real_times = {}
    for task_type, path in REAL_GROUP_TIME_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing real profile file for {task_type}: {path}. "
                "Run examples/hybrid/profile.sh first, or set DISTRIBUTION to a synthetic distribution."
            )
        values = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                values.append(float(row["group_time_max"]) * REAL_GROUP_TIME_SCALE.get(task_type, 1.0))
        if not values:
            raise ValueError(f"Real profile file for {task_type} has no group_time_max values: {path}")
        real_times[task_type] = np.asarray(values, dtype=float)

    _REAL_GROUP_TIME_CACHE = real_times
    return real_times


def sample_real_group_time(task_type, rng):
    real_times = load_real_group_times()[task_type]
    return float(rng.choice(real_times))


def sample_single_time(mean, std, rng):
    """根据分布类型采样单个 sample 的推理时间。"""
    std = max(float(std), 1e-9)
    if DISTRIBUTION == "exponential":
        return rng.exponential(mean)
    if DISTRIBUTION == "gamma":
        # 用真实均值/方差反推 gamma 参数:
        # mean = shape * scale, variance = shape * scale^2
        shape = (mean / std) ** 2
        scale = (std * std) / mean
        return rng.gamma(shape, scale)
    if DISTRIBUTION == "lognormal":
        # 用真实均值/方差反推 lognormal 底层正态参数。
        sigma2 = np.log(1 + (std * std) / (mean * mean))
        sigma = np.sqrt(sigma2)
        mu = np.log(mean) - sigma2 / 2
        return rng.lognormal(mu, sigma)
    if DISTRIBUTION == "pareto":
        xm = mean * (PARETO_ALPHA - 1) / PARETO_ALPHA
        return (rng.pareto(PARETO_ALPHA) + 1) * xm
    if DISTRIBUTION == "uniform":
        return rng.uniform(0, 2 * mean)
    if DISTRIBUTION == "constant":
        return mean
    return rng.exponential(mean)


def sample_group_time(mean, std, rng, task_type=None):
    """采样一个 group 的完成时间。

    synthetic 分布下，group 内 N_SAMPLES_PER_PROMPT 个 sample 并行推理，
    全部完成才算 group 完成。real 分布直接采样 profile 到的 group_time_max。
    """
    if DISTRIBUTION == "real":
        if task_type is None:
            raise ValueError("task_type is required when DISTRIBUTION=real")
        return sample_real_group_time(task_type, rng)
    return max(sample_single_time(mean, std, rng) for _ in range(N_SAMPLES_PER_PROMPT))


# ========== 辅助函数 ==========
def get_lag_sample_for_group(task_type, dispatch_batch, batch_history, cur_batch_num):
    """和 off-policy mask 使用同一口径计算单个 in-flight group 的 lag_sample。

    dispatch_batch 到 cur_batch_num 之间，已经训练过多少同 task sample，
    这个 group 就落后多少同 task sample。
    """
    if dispatch_batch is None or dispatch_batch >= cur_batch_num:
        return 0

    task_idx = 0 if task_type == "math" else 1
    lag_sample = 0
    for batch_idx in range(dispatch_batch, cur_batch_num):
        if 0 <= batch_idx < len(batch_history):
            lag_sample += batch_history[batch_idx][task_idx]
    return lag_sample



def get_completed_batch_lag_metrics(current_batch_samples, batch_history, cur_batch_num):
    """Match wandb tool/lag_sample_* metrics for the completed training batch.

    For each completed group in the batch, compute lag_sample against the same-task
    samples trained between dispatch_batch and cur_batch_num, then average/max by task.
    """
    math_lags = []
    qa_lags = []
    for task_type, _, dispatch_batch in current_batch_samples:
        lag_sample = get_lag_sample_for_group(task_type, dispatch_batch, batch_history, cur_batch_num)
        if task_type == "math":
            math_lags.append(lag_sample)
        else:
            qa_lags.append(lag_sample)

    metrics = {
        "train_lag_sample_math_count": len(math_lags),
        "train_lag_sample_qa_count": len(qa_lags),
    }
    if math_lags:
        metrics["train_lag_sample_math_average"] = sum(math_lags) / len(math_lags)
        metrics["train_lag_sample_math_max"] = max(math_lags)
    else:
        metrics["train_lag_sample_math_average"] = 0.0
        metrics["train_lag_sample_math_max"] = 0
    if qa_lags:
        metrics["train_lag_sample_qa_average"] = sum(qa_lags) / len(qa_lags)
        metrics["train_lag_sample_qa_max"] = max(qa_lags)
    else:
        metrics["train_lag_sample_qa_average"] = 0.0
        metrics["train_lag_sample_qa_max"] = 0
    return metrics


def get_lag_metrics(in_flight, batch_history, cur_batch_num):
    """基于所有 in-flight group 累计 lag_sample，和当前生产代码保持一致。

    in_flight: dict[req_id -> (type, problem_id, dispatch_batch)]
    batch_history: list[(math_cnt, qa_cnt)]，每个已训练 batch 的 sample 计数
    cur_batch_num: 当前 policy/batch version，等价于 len(batch_history)

    注意：lag_sample_math/qa 不是按最老版本回溯一次，而是对每个
    in-flight group 计算自己的 lag_sample，再按 task 累加。
    """
    metrics = {
        "max_lag_math": 0,
        "max_lag_qa": 0,
        "lag_sample_math": 0,
        "lag_sample_qa": 0,
        "in_flight_math": 0,
        "in_flight_qa": 0,
    }

    for task_type, _, dispatch_batch in in_flight.values():
        version_lag = max(0, cur_batch_num - dispatch_batch)
        group_lag_sample = get_lag_sample_for_group(
            task_type, dispatch_batch, batch_history, cur_batch_num
        )
        if task_type == "math":
            metrics["in_flight_math"] += 1
            metrics["max_lag_math"] = max(metrics["max_lag_math"], version_lag)
            metrics["lag_sample_math"] += group_lag_sample
        else:
            metrics["in_flight_qa"] += 1
            metrics["max_lag_qa"] = max(metrics["max_lag_qa"], version_lag)
            metrics["lag_sample_qa"] += group_lag_sample

    return metrics


def get_dynamic_math_prob(lag_sample_math, lag_sample_qa, cur_batch_num):
    """模拟 custom_data_source.py 当前 dynamic alternation 的概率计算。"""
    base_math_prob = MATH_RATIO
    in_warmup = cur_batch_num < WARMUP_STEPS
    if in_warmup:
        return {
            "base_math_prob": base_math_prob,
            "lag_based_math_prob": base_math_prob,
            "alpha": 0.0,
            "math_prob": base_math_prob,
            "in_warmup": True,
        }

    if lag_sample_math == 0 and lag_sample_qa == 0:
        lag_based_math_prob = base_math_prob
    else:
        lag_based_math_prob = lag_sample_qa / (lag_sample_math + lag_sample_qa)

    alpha = min(1.0, max(0.0, DYNAMIC_ALPHA))
    min_math_prob = min(MIN_MATH_RATIO, MAX_MATH_RATIO)
    max_math_prob = max(MIN_MATH_RATIO, MAX_MATH_RATIO)
    smoothed_math_prob = (1.0 - alpha) * base_math_prob + alpha * lag_based_math_prob
    math_prob = min(max_math_prob, max(min_math_prob, smoothed_math_prob))

    return {
        "base_math_prob": base_math_prob,
        "lag_based_math_prob": lag_based_math_prob,
        "alpha": alpha,
        "math_prob": math_prob,
        "in_warmup": False,
    }


def choose_next_type_dynamic(lag_sample_math, lag_sample_qa, cur_batch_num, rng):
    """dynamic_alternation: 反比概率 + math_ratio 平滑 + min/max clamp + warmup。"""
    prob_info = get_dynamic_math_prob(lag_sample_math, lag_sample_qa, cur_batch_num)
    task_type = "math" if rng.random() < prob_info["math_prob"] else "qa"
    return task_type, prob_info


def choose_next_type_fixed(math_ratio, rng):
    """fixed_ratio: 按固定 math_ratio 决定下一请求的类型。"""
    return "math" if rng.random() < math_ratio else "qa"


def choose_next_type_block(dispatched_count):
    """fixed block alternation: 连续 BLOCK_ALTERNATION_SIZE 个 math，再连续同样数量 qa。"""
    block_idx = dispatched_count // BLOCK_ALTERNATION_SIZE
    return "math" if block_idx % 2 == 0 else "qa"


def _other_task(task_type):
    return "qa" if task_type == "math" else "math"


def choose_next_type_drr(dispatched_math, dispatched_count):
    """Deficit round-robin: 选择下一条后，让全局 math 比例尽量贴近目标比例。"""
    target_math_after_next = (dispatched_count + 1) * MATH_RATIO
    return "math" if dispatched_math < target_math_after_next else "qa"


def apply_consecutive_cap(task_type, consecutive_type, consecutive_count):
    if consecutive_type == task_type and consecutive_count >= MAX_CONSECUTIVE_DISPATCH:
        return _other_task(task_type)
    return task_type


def choose_next_type_drr_cap(dispatched_math, dispatched_count, consecutive_type, consecutive_count):
    task_type = choose_next_type_drr(dispatched_math, dispatched_count)
    return apply_consecutive_cap(task_type, consecutive_type, consecutive_count)


def choose_next_type_window(dispatch_history, dispatched_math, dispatched_count):
    """Windowed ratio control: 优先控制最近 WINDOW_RATIO_SIZE 个 dispatch 的 math 比例。"""
    if len(dispatch_history) < WINDOW_RATIO_SIZE:
        return choose_next_type_drr(dispatched_math, dispatched_count)

    recent = dispatch_history[-WINDOW_RATIO_SIZE:]
    recent_math = sum(1 for task_type in recent if task_type == "math")
    target_math = WINDOW_RATIO_SIZE * MATH_RATIO
    if recent_math < target_math:
        return "math"
    if recent_math > target_math:
        return "qa"
    return choose_next_type_drr(dispatched_math, dispatched_count)


def choose_next_type_debt_lag(dispatched_math, dispatched_count, lag_sample_math, lag_sample_qa):
    """Ratio debt + lag penalty: 长期比例靠 debt 拉回，当前 stale 靠 lag penalty 避免继续加压。"""
    lag_normalizer = max(1.0, PARALLEL * N_SAMPLES_PER_PROMPT)
    best_task = None
    best_cost = None
    for task_type in ("math", "qa"):
        projected_math = dispatched_math + (1 if task_type == "math" else 0)
        target_math = (dispatched_count + 1) * MATH_RATIO
        ratio_cost = abs(projected_math - target_math)
        lag_sample = lag_sample_math if task_type == "math" else lag_sample_qa
        lag_cost = LAG_PENALTY_BETA * lag_sample / lag_normalizer
        cost = ratio_cost + lag_cost
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_task = task_type
    return best_task


def choose_next_type_debt_lag_cap(
    dispatched_math, dispatched_count, lag_sample_math, lag_sample_qa, consecutive_type, consecutive_count
):
    task_type = choose_next_type_debt_lag(dispatched_math, dispatched_count, lag_sample_math, lag_sample_qa)
    return apply_consecutive_cap(task_type, consecutive_type, consecutive_count)


def choose_initial_type(mode, dispatched_count, dispatched_math, dispatch_history, consecutive_type, consecutive_count, rng):
    if mode == "block":
        return choose_next_type_block(dispatched_count)
    if mode == "drr":
        return choose_next_type_drr(dispatched_math, dispatched_count)
    if mode == "drr_cap":
        return choose_next_type_drr_cap(dispatched_math, dispatched_count, consecutive_type, consecutive_count)
    if mode == "window":
        return choose_next_type_window(dispatch_history, dispatched_math, dispatched_count)
    if mode in ("debt_lag", "debt_lag_cap"):
        return choose_next_type_drr(dispatched_math, dispatched_count)
    return choose_next_type_fixed(MATH_RATIO, rng)


def choose_dispatch_type(
    mode,
    dispatched_count,
    dispatched_math,
    dispatch_history,
    consecutive_type,
    consecutive_count,
    in_flight,
    batch_history,
    cur_batch_num,
    rng,
):
    """Choose one newly dispatched group type for both simulation engines."""
    if mode == "dynamic":
        metrics = get_lag_metrics(in_flight, batch_history, cur_batch_num)
        typ, _ = choose_next_type_dynamic(
            metrics["lag_sample_math"], metrics["lag_sample_qa"], cur_batch_num, rng
        )
    elif mode == "block":
        typ = choose_next_type_block(dispatched_count)
    elif mode == "drr":
        typ = choose_next_type_drr(dispatched_math, dispatched_count)
    elif mode == "drr_cap":
        typ = choose_next_type_drr_cap(
            dispatched_math, dispatched_count, consecutive_type, consecutive_count
        )
    elif mode == "window":
        typ = choose_next_type_window(dispatch_history, dispatched_math, dispatched_count)
    elif mode in ("debt_lag", "debt_lag_cap"):
        metrics = get_lag_metrics(in_flight, batch_history, cur_batch_num)
        if mode == "debt_lag":
            typ = choose_next_type_debt_lag(
                dispatched_math, dispatched_count, metrics["lag_sample_math"], metrics["lag_sample_qa"]
            )
        else:
            typ = choose_next_type_debt_lag_cap(
                dispatched_math,
                dispatched_count,
                metrics["lag_sample_math"],
                metrics["lag_sample_qa"],
                consecutive_type,
                consecutive_count,
            )
    else:
        typ = choose_next_type_fixed(MATH_RATIO, rng)
    return typ


def sample_problem_and_duration(typ, rng):
    if typ == "math":
        prob_id = rng.integers(0, NUM_MATH_PROBLEMS)
        duration = sample_group_time(MATH_MEAN_TIME, MATH_STD_TIME, rng, "math")
    else:
        prob_id = rng.integers(0, NUM_QA_PROBLEMS)
        duration = sample_group_time(QA_MEAN_TIME, QA_STD_TIME, rng, "qa")
    return prob_id, duration


def get_train_time_for_batch(batch_idx):
    if (
        TRAIN_TIME_AFTER_BATCH is not None
        and TRAIN_TIME_AFTER_VALUE is not None
        and batch_idx >= TRAIN_TIME_AFTER_BATCH
    ):
        return TRAIN_TIME_AFTER_VALUE
    return TRAIN_TIME_PER_BATCH


def run_simulation(mode, seed=42):
    """运行一次模拟。

    mode: "dynamic", "fixed", "block", "drr", "drr_cap", "window", "debt_lag" 或 "debt_lag_cap"
    返回: batch_history, per_batch_metrics
    """
    rng = np.random.default_rng(seed)

    batch_history = []
    current_batch_samples = []       # (type, problem_id, dispatch_batch_num)
    per_batch_metrics = []

    # in-flight tracking: {req_id: (type, problem_id, dispatch_batch)}
    in_flight = {}
    cur_dispatch_batch = 0

    heap = []
    next_req_id = 0
    total_samples_needed = NUM_BATCHES * PARALLEL
    completed_count = 0
    dispatched_count = 0
    dispatched_math = 0
    dispatch_history = []
    consecutive_type = None
    consecutive_count = 0

    # 启动前 PARALLEL 个请求（初始 batch，无 lag 信息）
    for _ in range(PARALLEL):
        typ = choose_initial_type(
            mode, dispatched_count, dispatched_math, dispatch_history, consecutive_type, consecutive_count, rng
        )
        if typ == "math":
            prob_id = rng.integers(0, NUM_MATH_PROBLEMS)
            finish_time = sample_group_time(MATH_MEAN_TIME, MATH_STD_TIME, rng, "math")
        else:
            prob_id = rng.integers(0, NUM_QA_PROBLEMS)
            finish_time = sample_group_time(QA_MEAN_TIME, QA_STD_TIME, rng, "qa")
        heapq.heappush(heap, (finish_time, next_req_id, typ, prob_id, cur_dispatch_batch))
        in_flight[next_req_id] = (typ, prob_id, cur_dispatch_batch)
        next_req_id += 1
        dispatched_count += 1
        dispatched_math += 1 if typ == "math" else 0
        dispatch_history.append(typ)
        if consecutive_type == typ:
            consecutive_count += 1
        else:
            consecutive_type = typ
            consecutive_count = 1

    # 主模拟循环
    while completed_count < total_samples_needed:
        finish_time, req_id, typ, prob_id, disp_batch = heapq.heappop(heap)

        del in_flight[req_id]
        current_batch_samples.append((typ, prob_id, disp_batch))
        completed_count += 1

        # batch 已满，模拟 trainer 消费一个 batch，policy/batch version +1
        if len(current_batch_samples) == PARALLEL:
            math_group_cnt = sum(1 for t, _, _ in current_batch_samples if t == "math")
            qa_group_cnt = PARALLEL - math_group_cnt
            math_cnt = math_group_cnt * N_SAMPLES_PER_PROMPT
            qa_cnt = qa_group_cnt * N_SAMPLES_PER_PROMPT

            # 记录 append 之前的调度状态；此时 batch_history 还不包含当前 batch。
            cur_batch_num = len(batch_history)
            metrics = get_lag_metrics(in_flight, batch_history, cur_batch_num)
            metrics["inflight_lag_sample_math"] = metrics["lag_sample_math"]
            metrics["inflight_lag_sample_qa"] = metrics["lag_sample_qa"]
            metrics.update(get_completed_batch_lag_metrics(current_batch_samples, batch_history, cur_batch_num))
            metrics.update(
                get_dynamic_math_prob(
                    metrics["lag_sample_math"], metrics["lag_sample_qa"], cur_batch_num
                )
            )
            per_batch_metrics.append(metrics)

            batch_history.append((math_cnt, qa_cnt))
            current_batch_samples = []

        # 派发新请求（保持并行数恒定）
        if completed_count < total_samples_needed:
            cur_batch_num = len(batch_history)

            metrics = None
            if mode == "dynamic":
                metrics = get_lag_metrics(in_flight, batch_history, cur_batch_num)
                typ, _ = choose_next_type_dynamic(
                    metrics["lag_sample_math"], metrics["lag_sample_qa"], cur_batch_num, rng
                )
            elif mode == "block":
                typ = choose_next_type_block(dispatched_count)
            elif mode == "drr":
                typ = choose_next_type_drr(dispatched_math, dispatched_count)
            elif mode == "drr_cap":
                typ = choose_next_type_drr_cap(
                    dispatched_math, dispatched_count, consecutive_type, consecutive_count
                )
            elif mode == "window":
                typ = choose_next_type_window(dispatch_history, dispatched_math, dispatched_count)
            elif mode in ("debt_lag", "debt_lag_cap"):
                metrics = get_lag_metrics(in_flight, batch_history, cur_batch_num)
                if mode == "debt_lag":
                    typ = choose_next_type_debt_lag(
                        dispatched_math, dispatched_count, metrics["lag_sample_math"], metrics["lag_sample_qa"]
                    )
                else:
                    typ = choose_next_type_debt_lag_cap(
                        dispatched_math,
                        dispatched_count,
                        metrics["lag_sample_math"],
                        metrics["lag_sample_qa"],
                        consecutive_type,
                        consecutive_count,
                    )
            else:
                typ = choose_next_type_fixed(MATH_RATIO, rng)

            if typ == "math":
                prob_id = rng.integers(0, NUM_MATH_PROBLEMS)
                duration = sample_group_time(MATH_MEAN_TIME, MATH_STD_TIME, rng, "math")
            else:
                prob_id = rng.integers(0, NUM_QA_PROBLEMS)
                duration = sample_group_time(QA_MEAN_TIME, QA_STD_TIME, rng, "qa")

            heapq.heappush(heap, (finish_time + duration, next_req_id, typ, prob_id, cur_batch_num))
            in_flight[next_req_id] = (typ, prob_id, cur_batch_num)
            next_req_id += 1
            dispatched_count += 1
            dispatched_math += 1 if typ == "math" else 0
            dispatch_history.append(typ)
            if consecutive_type == typ:
                consecutive_count += 1
            else:
                consecutive_type = typ
                consecutive_count = 1

    return batch_history, per_batch_metrics, dispatch_history


def run_simulation_buffered(mode, seed=42):
    """Run a trainer/rollout simulation with an explicit ready buffer.

    This is closer to the hybrid async path than ``run_simulation``:
    rollout completions enter ``ready_buffer`` while train is busy, and each
    train step consumes ready samples before waiting for fresh completions.
    """
    rng = np.random.default_rng(seed)

    batch_history = []
    per_batch_metrics = []
    dispatch_history = []
    in_flight = {}
    ready_buffer = []
    heap = []

    next_req_id = 0
    dispatched_count = 0
    dispatched_math = 0
    consecutive_type = None
    consecutive_count = 0
    current_time = 0.0
    buffer_seq = 0

    def record_dispatch_type(typ):
        nonlocal dispatched_count, dispatched_math, consecutive_type, consecutive_count
        dispatched_count += 1
        dispatched_math += 1 if typ == "math" else 0
        dispatch_history.append(typ)
        if consecutive_type == typ:
            consecutive_count += 1
        else:
            consecutive_type = typ
            consecutive_count = 1

    def dispatch_one(start_time, dispatch_batch):
        nonlocal next_req_id
        typ = choose_dispatch_type(
            mode,
            dispatched_count,
            dispatched_math,
            dispatch_history,
            consecutive_type,
            consecutive_count,
            in_flight,
            batch_history,
            dispatch_batch,
            rng,
        )
        prob_id, duration = sample_problem_and_duration(typ, rng)
        finish_time = start_time + duration
        heapq.heappush(heap, (finish_time, next_req_id, typ, prob_id, dispatch_batch))
        in_flight[next_req_id] = (typ, prob_id, dispatch_batch)
        next_req_id += 1
        record_dispatch_type(typ)

    def complete_one(dispatch_batch):
        nonlocal current_time, buffer_seq
        finish_time, req_id, typ, prob_id, disp_batch = heapq.heappop(heap)
        current_time = max(current_time, finish_time)
        del in_flight[req_id]
        heapq.heappush(ready_buffer, (finish_time, buffer_seq, typ, prob_id, disp_batch))
        buffer_seq += 1
        dispatch_one(finish_time, dispatch_batch)

    def drain_until(end_time, dispatch_batch):
        while heap and heap[0][0] <= end_time:
            complete_one(dispatch_batch)

    # 启动前保持 PARALLEL 个 in-flight group。
    for _ in range(PARALLEL):
        dispatch_one(0.0, 0)

    while len(batch_history) < NUM_BATCHES:
        cur_batch_num = len(batch_history)

        # 训练开始前，先消费已经完成的 buffer；如果不够，再等待新的 rollout 完成。
        drain_until(current_time, cur_batch_num)
        while len(ready_buffer) < PARALLEL:
            complete_one(cur_batch_num)

        ready_buffer_size_before_pop = len(ready_buffer)
        ready_buffer_math_before_pop = sum(1 for _, _, typ, _, _ in ready_buffer if typ == "math")
        ready_buffer_qa_before_pop = sum(1 for _, _, typ, _, _ in ready_buffer if typ == "qa")

        current_batch_samples = []
        for _ in range(PARALLEL):
            _, _, typ, prob_id, disp_batch = heapq.heappop(ready_buffer)
            current_batch_samples.append((typ, prob_id, disp_batch))

        math_group_cnt = sum(1 for t, _, _ in current_batch_samples if t == "math")
        qa_group_cnt = PARALLEL - math_group_cnt
        math_cnt = math_group_cnt * N_SAMPLES_PER_PROMPT
        qa_cnt = qa_group_cnt * N_SAMPLES_PER_PROMPT

        metrics = get_lag_metrics(in_flight, batch_history, cur_batch_num)
        metrics["inflight_lag_sample_math"] = metrics["lag_sample_math"]
        metrics["inflight_lag_sample_qa"] = metrics["lag_sample_qa"]
        metrics.update(get_completed_batch_lag_metrics(current_batch_samples, batch_history, cur_batch_num))
        metrics.update(get_dynamic_math_prob(metrics["lag_sample_math"], metrics["lag_sample_qa"], cur_batch_num))
        metrics["ready_buffer_size_after_pop"] = len(ready_buffer)
        metrics["ready_buffer_math_after_pop"] = sum(1 for _, _, typ, _, _ in ready_buffer if typ == "math")
        metrics["ready_buffer_qa_after_pop"] = sum(1 for _, _, typ, _, _ in ready_buffer if typ == "qa")
        metrics["ready_buffer_size_before_pop"] = ready_buffer_size_before_pop
        metrics["ready_buffer_math_before_pop"] = ready_buffer_math_before_pop
        metrics["ready_buffer_qa_before_pop"] = ready_buffer_qa_before_pop
        train_time = get_train_time_for_batch(cur_batch_num)
        metrics["train_time"] = train_time
        per_batch_metrics.append(metrics)

        # 训练期间 rollout 继续跑，完成的样本进入 ready_buffer；新 dispatch 仍使用当前 policy version。
        train_end_time = current_time + train_time
        drain_until(train_end_time, cur_batch_num)
        current_time = train_end_time

        batch_history.append((math_cnt, qa_cnt))

    return batch_history, per_batch_metrics, dispatch_history


def print_results(mode_name, batch_history, per_batch_metrics):
    print(f"\n{'=' * 75}")
    print(f"  模式: {mode_name}")
    print(f"{'=' * 130}")
    print(
        f"共 {len(batch_history)} 个 batch，每 batch {PARALLEL} 个 group，"
        f"每 group {N_SAMPLES_PER_PROMPT} 个 sample"
    )

    print(
        f"\n{'Batch':>6} | {'math_g':>6} : {'qa_g':>6} | {'math%':>6} | "
        f"{'if_m':>4} {'if_q':>4} | {'max_lag_m':>9} {'max_lag_q':>9} | "
        f"{'lag_samp_m':>10} {'lag_samp_q':>10} | {'lag_p':>6} {'math_p':>6} {'warm':>4}"
    )
    print("-" * 130)

    for i, ((m_samples, q_samples), metrics) in enumerate(zip(batch_history, per_batch_metrics)):
        m_groups = m_samples // N_SAMPLES_PER_PROMPT
        q_groups = q_samples // N_SAMPLES_PER_PROMPT
        ratio = m_samples / (m_samples + q_samples)
        lag_p = metrics.get("lag_based_math_prob", MATH_RATIO)
        math_p = metrics.get("math_prob", MATH_RATIO)
        warm = "yes" if metrics.get("in_warmup", False) else "no"
        print(
            f"B{i + 1:3d}  | {m_groups:6d} : {q_groups:6d} | {ratio:5.1%} | "
            f"{metrics['in_flight_math']:4d} {metrics['in_flight_qa']:4d} | "
            f"{metrics['max_lag_math']:9d} {metrics['max_lag_qa']:9d} | "
            f"{metrics['lag_sample_math']:10d} {metrics['lag_sample_qa']:10d} | "
            f"{lag_p:6.3f} {math_p:6.3f} {warm:>4}"
        )

    total_math = sum(m for m, _ in batch_history)
    total_qa = sum(q for _, q in batch_history)
    print(
        f"\n总体: math_samples={total_math}, qa_samples={total_qa}, "
        f"math比例={total_math / (total_math + total_qa):.3f}"
    )

    all_lm = [m["max_lag_math"] for m in per_batch_metrics]
    all_lq = [m["max_lag_qa"] for m in per_batch_metrics]
    all_sm = [m["train_lag_sample_math_average"] for m in per_batch_metrics]
    all_sq = [m["train_lag_sample_qa_average"] for m in per_batch_metrics]
    all_lbp = [m.get("lag_based_math_prob", MATH_RATIO) for m in per_batch_metrics]
    all_mp = [m.get("math_prob", MATH_RATIO) for m in per_batch_metrics]
    print("\nlag 指标汇总:")
    print(f"  max_lag_math    — 均值: {np.mean(all_lm):.2f}, 最大: {np.max(all_lm)}, 中位数: {np.median(all_lm):.0f}")
    print(f"  max_lag_qa      — 均值: {np.mean(all_lq):.2f}, 最大: {np.max(all_lq)}, 中位数: {np.median(all_lq):.0f}")
    print(f"  train_lag_sample_math_average — 均值: {np.mean(all_sm):.1f}, 最大: {np.max(all_sm)}, 最小: {np.min(all_sm)}")
    print(f"  train_lag_sample_qa_average   — 均值: {np.mean(all_sq):.1f}, 最大: {np.max(all_sq)}, 最小: {np.min(all_sq)}")
    print(f"  lag_based_p     — 均值: {np.mean(all_lbp):.3f}, 最大: {np.max(all_lbp):.3f}, 最小: {np.min(all_lbp):.3f}")
    print(f"  final_math_prob — 均值: {np.mean(all_mp):.3f}, 最大: {np.max(all_mp):.3f}, 最小: {np.min(all_mp):.3f}")


def print_compact_comparison(title, runs_by_mode):
    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")
    print(
        f"{'mode':<12} {'steps':>9} {'mavg':>8} {'mp90':>8} {'mcnt':>8} "
        f"{'qavg':>8} {'buf_pre':>8} {'buf_m':>8} {'buf_left':>8}"
    )
    print("-" * 90)
    for mode_name, runs in runs_by_mode.items():
        for start, end in ((0, 180), (180, 220), (220, 260), (260, 300)):
            vals = [metrics[i] for metrics in runs for i in range(start, min(end, len(metrics)))]
            if not vals:
                continue
            mavg = np.mean([m["train_lag_sample_math_average"] for m in vals])
            mp90 = np.percentile([m["train_lag_sample_math_average"] for m in vals], 90)
            mcnt = np.mean([m["train_lag_sample_math_count"] * N_SAMPLES_PER_PROMPT for m in vals])
            qavg = np.mean([m["train_lag_sample_qa_average"] for m in vals])
            buf = np.mean([m.get("ready_buffer_size_before_pop", 0) for m in vals])
            buf_m = np.mean([m.get("ready_buffer_math_before_pop", 0) for m in vals])
            buf_left = np.mean([m.get("ready_buffer_size_after_pop", 0) for m in vals])
            print(
                f"{mode_name:<12} {start:>3d}-{end:<3d} "
                f"{mavg:8.1f} {mp90:8.1f} {mcnt:8.1f} {qavg:8.1f} "
                f"{buf:8.1f} {buf_m:8.1f} {buf_left:8.1f}"
            )


def run_many(sim_fn, mode, seeds=range(5)):
    runs = []
    for seed in seeds:
        _, metrics, _ = sim_fn(mode, seed=seed)
        runs.append(metrics)
    return runs


# ========== 运行两种模式 ==========
if __name__ == "__main__":
    print(f"分布: {DISTRIBUTION}")
    print(
        f"dynamic 参数: math_ratio={MATH_RATIO}, alpha={DYNAMIC_ALPHA}, "
        f"min_math_ratio={MIN_MATH_RATIO}, max_math_ratio={MAX_MATH_RATIO}, "
        f"warmup_steps={WARMUP_STEPS}, block_size={BLOCK_ALTERNATION_SIZE}"
    )

    print_compact_comparison(
        "heap-only baseline: no train-time ready buffer",
        {
            "fixed": run_many(run_simulation, "fixed"),
            "block16": run_many(run_simulation, "block"),
        },
    )

    TRAIN_TIME_PER_BATCH = 0.0
    TRAIN_TIME_AFTER_BATCH = 180
    TRAIN_TIME_AFTER_VALUE = 40.0
    print_compact_comparison(
        "buffered simulation: train_time jumps from 0 to 40 at step 180",
        {
            "fixed": run_many(run_simulation_buffered, "fixed"),
            "block16": run_many(run_simulation_buffered, "block"),
        },
    )
