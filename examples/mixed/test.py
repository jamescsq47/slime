import heapq
import json
import warnings
from pathlib import Path

import numpy as np

# ========== 参数 ==========
NUM_MATH_PROBLEMS = 100      # math 类问题个数
NUM_QA_PROBLEMS = 100        # qa 类问题个数
PARALLEL = 32               # 同时推理的 group 数（rollout batch size）
NUM_BATCHES = 200           # 模拟的 batch 数量
N_SAMPLES_PER_PROMPT = 8    # 每个 group 内的 sample 数（n_samples_per_prompt）
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
    "math": Path(__file__).resolve().parent.parent / "hybrid" / "debug" / "profile" / "profile_math_groups8_rollout0.jsonl",
    "qa": REAL_PROFILE_DIR.parent / "browsecomp_group_max_times_0_192.jsonl",
}
REAL_GROUP_TIME_SCALE = {
    "math": 1.0,
    "qa": 1.0,
}
# pareto 参数 (shape alpha) — alpha 越小尾部越重
PARETO_ALPHA = 2.0


_REAL_GROUP_TIME_CACHE = None


def load_real_group_times():
    """Load measured group completion times from profile.sh output.

    Missing task files are allowed so BrowseComp can use measured group-max
    times while math falls back to the synthetic distribution.
    """
    global _REAL_GROUP_TIME_CACHE
    if _REAL_GROUP_TIME_CACHE is not None:
        return _REAL_GROUP_TIME_CACHE

    real_times = {}
    for task_type, path in REAL_GROUP_TIME_FILES.items():
        if not path.exists():
            warnings.warn(
                f"Missing real profile file for {task_type}: {path}. "
                "This task will fall back to synthetic group times.",
                RuntimeWarning,
            )
            continue
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
    real_times = load_real_group_times()
    if task_type in real_times:
        return float(rng.choice(real_times[task_type]))
    if task_type == "math":
        return max(sample_single_time(MATH_MEAN_TIME, MATH_STD_TIME, rng) for _ in range(N_SAMPLES_PER_PROMPT))
    return max(sample_single_time(QA_MEAN_TIME, QA_STD_TIME, rng) for _ in range(N_SAMPLES_PER_PROMPT))


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
    all_sm = [m["lag_sample_math"] for m in per_batch_metrics]
    all_sq = [m["lag_sample_qa"] for m in per_batch_metrics]
    all_lbp = [m.get("lag_based_math_prob", MATH_RATIO) for m in per_batch_metrics]
    all_mp = [m.get("math_prob", MATH_RATIO) for m in per_batch_metrics]
    print("\nlag 指标汇总:")
    print(f"  max_lag_math    — 均值: {np.mean(all_lm):.2f}, 最大: {np.max(all_lm)}, 中位数: {np.median(all_lm):.0f}")
    print(f"  max_lag_qa      — 均值: {np.mean(all_lq):.2f}, 最大: {np.max(all_lq)}, 中位数: {np.median(all_lq):.0f}")
    print(f"  lag_sample_math — 均值: {np.mean(all_sm):.1f}, 最大: {np.max(all_sm)}, 最小: {np.min(all_sm)}")
    print(f"  lag_sample_qa   — 均值: {np.mean(all_sq):.1f}, 最大: {np.max(all_sq)}, 最小: {np.min(all_sq)}")
    print(f"  lag_based_p     — 均值: {np.mean(all_lbp):.3f}, 最大: {np.max(all_lbp):.3f}, 最小: {np.min(all_lbp):.3f}")
    print(f"  final_math_prob — 均值: {np.mean(all_mp):.3f}, 最大: {np.max(all_mp):.3f}, 最小: {np.min(all_mp):.3f}")


# ========== 运行两种模式 ==========
if __name__ == "__main__":
    print(f"分布: {DISTRIBUTION}")
    print(
        f"dynamic 参数: math_ratio={MATH_RATIO}, alpha={DYNAMIC_ALPHA}, "
        f"min_math_ratio={MIN_MATH_RATIO}, max_math_ratio={MAX_MATH_RATIO}, "
        f"warmup_steps={WARMUP_STEPS}, block_size={BLOCK_ALTERNATION_SIZE}"
    )

    h1, m1, _ = run_simulation("dynamic", seed=42)
    print_results("dynamic (all in-flight lag + smooth/clamp/warmup)", h1, m1)

    h2, m2, _ = run_simulation("fixed", seed=42)
    print_results(f"fixed (math_ratio={MATH_RATIO})", h2, m2)

    h3, m3, _ = run_simulation("block", seed=42)
    print_results(f"block alternation ({BLOCK_ALTERNATION_SIZE} math / {BLOCK_ALTERNATION_SIZE} qa dispatch)", h3, m3)
