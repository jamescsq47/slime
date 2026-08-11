"""Custom DataSource for unified math and BrowseComp data loading in slime framework.

Behavior is identical to the default RolloutDataSourceWithBuffer for math data
and the BrowseComp data loading for QA data, with the only addition being
extra metadata fields (task_type, tools_available) on each Sample.
"""

import copy
import logging
import math
import os
import random

import torch
from pathlib import Path

from slime.rollout.data_source import DataSource
from slime.utils.data import Dataset
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class CustomDataSource(DataSource):
    """统一的数据源，支持 math、QA 和 Terminal-Bench 混合加载。"""
    
    def __init__(self, args):
        """
        Args:
            args: 包含以下参数：
                - hf_checkpoint: huggingface模型检查点
                - apply_chat_template: 是否应用聊天模板
                - apply_chat_template_kwargs: 聊天模板参数
                - rollout_shuffle: 是否打乱数据
                - rollout_seed: 随机种子
                - batch_alternation: 是否启用batch-level交替模式 (default: False)
                - math_batches_per_cycle: 每个周期中math的batch数 (default: 1)
                - qa_batches_per_cycle: 每个周期中QA的batch数 (default: 1)
                
        需要从args或环境中设置：
            - math_data_path: 数学数据路径
            - qa_data_path: BrowseComp数据路径
            - math_ratio: math数据的比例（默认0.7，仅在batch_alternation=False时使用）
        """
        self.args = args
        self.epoch_id = 0
        self.sample_offset = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.metadata = {}
        self.origin_samples = []
        self.samples = []
        self.buffer = []
        
        # 获取混合比例参数
        self.math_ratio = getattr(args, 'math_ratio', 0.5)
        self.terminal_ratio = getattr(args, 'terminal_ratio', 0.0)
        if self.math_ratio < 0 or self.terminal_ratio < 0 or self.math_ratio + self.terminal_ratio > 1:
            raise ValueError(
                "math_ratio and terminal_ratio must be non-negative and sum to at most 1; "
                f"got math_ratio={self.math_ratio}, terminal_ratio={self.terminal_ratio}"
            )
        
        # 获取batch-level交替参数
        self.batch_alternation = getattr(args, 'batch_alternation', False)
        self.count_aware_alternation = getattr(args, 'count_aware_alternation', False)
        self.math_batches_per_cycle = getattr(args, 'math_batches_per_cycle', 1)
        self.qa_batches_per_cycle = getattr(args, 'qa_batches_per_cycle', 1)
        self.batch_alternation_start_task = getattr(args, 'batch_alternation_start_task', 'math')
        
        # 获取dynamic交替参数
        self.dynamic_alternation = getattr(args, 'dynamic_alternation', False)
        self.lag_version = getattr(args, 'lag_version', 5)
        self.dynamic_alternation_alpha = getattr(args, 'dynamic_alternation_alpha', 0.5)
        self.dynamic_alternation_warmup_steps = getattr(args, 'dynamic_alternation_warmup_steps', 5)
        self.dynamic_alternation_min_math_ratio = getattr(args, 'dynamic_alternation_min_math_ratio', 0.3)
        self.dynamic_alternation_max_math_ratio = getattr(args, 'dynamic_alternation_max_math_ratio', 0.7)
        if self.terminal_ratio > 0 and (
            self.batch_alternation or self.count_aware_alternation or self.dynamic_alternation
        ):
            raise ValueError(
                "Terminal-Bench currently supports normal mixed dispatch only; "
                "disable batch/count-aware/dynamic alternation when terminal_ratio > 0"
            )
        if self.terminal_ratio > 0 and (
            getattr(args, "train_batch_math_groups", None) is not None
            or getattr(args, "train_batch_qa_groups", None) is not None
        ):
            raise ValueError(
                "The Math/QA-only training quotas cannot admit Terminal groups; "
                "disable --train-batch-math-groups/--train-batch-qa-groups"
            )
        
        if args.rollout_global_dataset:
            # 加载tokenizer和processor
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
            
            # 备份逻辑，防止外部组件读取 dump 目录报错
            if getattr(args, 'dump_details', None) is not None:
                dump_path = Path(args.dump_details)
                tokenizer.save_pretrained(dump_path / "tokenizer")
                if processor:
                    processor.save_pretrained(dump_path / "processor")
            
            # 根据模式加载数据
            if self.dynamic_alternation:
                self._load_data_for_dynamic_alternation(args, tokenizer, processor)
            elif self.batch_alternation or self.count_aware_alternation:
                self._load_data_for_batch_alternation(args, tokenizer, processor)
            else:
                self._load_unified_data(args, tokenizer, processor)
            
            if self.args.rollout_shuffle and not self.batch_alternation and not self.count_aware_alternation and not self.dynamic_alternation:
                # 只在普通模式下shuffle，其他模式有自己的逻辑
                self.shuffle(self.epoch_id)
    
    @property
    def dataset(self):
        """返回一个兼容的dataset对象，提供len()支持"""
        class DatasetWrapper:
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
        
        return DatasetWrapper(self.origin_samples)
    
    def __len__(self):
        return len(self.dataset)
    
    def _load_math_data(self, args, tokenizer, processor, math_path):
        """加载数学数据，使用标准Dataset类，行为与RolloutDataSource完全一致"""
        if not math_path:
            return []

        dataset = Dataset(
            math_path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key,
            multimodal_keys=args.multimodal_keys,
            label_key=args.label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            seed=args.rollout_seed,
        )
        samples = dataset.origin_samples
        for s in samples:
            s.metadata = {**(s.metadata or {}), "task_type": "math", "tools_available": ["code_interpreter"]}
        logger.info(f"Loaded {len(samples)} math samples from {math_path}")
        return samples

    def _load_qa_data(self, args, tokenizer, processor, qa_path):
        """加载BrowseComp数据，保留message-list prompt给browsecomp_agent处理。"""
        if not qa_path:
            return []

        dataset = Dataset(
            qa_path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key,
            multimodal_keys=args.multimodal_keys,
            label_key=args.label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key,
            apply_chat_template=False,
            apply_chat_template_kwargs=None,
            seed=args.rollout_seed,
        )
        samples = dataset.origin_samples
        for s in samples:
            s.metadata = {
                **(s.metadata or {}),
                "task_type": "qa",
                "tools_available": ["search", "open_page", "finish"],
            }
        logger.info(f"Loaded {len(samples)} BrowseComp samples from {qa_path}")
        return samples

    def _load_terminal_data(self, args, tokenizer, processor, terminal_path):
        """Load Terminal-Bench task ids; reset() supplies the task instruction."""
        if not terminal_path:
            return []

        dataset = Dataset(
            terminal_path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key,
            multimodal_keys=args.multimodal_keys,
            # Terminal rows deliberately have no supervised label.
            label_key=None,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key,
            apply_chat_template=False,
            apply_chat_template_kwargs=None,
            seed=args.rollout_seed,
        )
        samples = dataset.origin_samples
        for sample in samples:
            metadata = sample.metadata or {}
            if not metadata.get("task_id") and not metadata.get("task_name"):
                raise ValueError("Every terminal sample must contain metadata.task_id")
            sample.metadata = {
                **metadata,
                "task_type": "terminal",
                "tools_available": ["shell"],
            }
        logger.info(f"Loaded {len(samples)} Terminal-Bench samples from {terminal_path}")
        return samples
    
    def _load_unified_data(self, args, tokenizer, processor):
        """加载并混合math和qa数据（普通模式）"""
        math_path = getattr(args, 'math_data_path', None)
        qa_path = getattr(args, 'qa_data_path', None)
        terminal_path = getattr(args, "terminal_data_path", None)
        ratios = {
            "math": float(self.math_ratio),
            "qa": float(1.0 - self.math_ratio - self.terminal_ratio),
            "terminal": float(self.terminal_ratio),
        }
        paths = {"math": math_path, "qa": qa_path, "terminal": terminal_path}
        for task_type, ratio in ratios.items():
            if ratio > 0 and not paths[task_type]:
                raise ValueError(f"{task_type}_data_path is required when {task_type}_ratio={ratio}")

        math_samples = self._load_math_data(args, tokenizer, processor, math_path) if ratios["math"] > 0 else []
        qa_samples = self._load_qa_data(args, tokenizer, processor, qa_path) if ratios["qa"] > 0 else []
        terminal_samples = (
            self._load_terminal_data(args, tokenizer, processor, terminal_path)
            if ratios["terminal"] > 0
            else []
        )
        if not math_samples and not qa_samples and not terminal_samples:
            raise ValueError("No mixed-domain data loaded")

        self.origin_samples = self._mix_samples(math_samples, qa_samples, terminal_samples)
        self.samples = self.origin_samples
        logger.info(
            f"Loaded {len(self.origin_samples)} total samples "
            f"(math source: {len(math_samples)}, qa source: {len(qa_samples)}, "
            f"terminal source: {len(terminal_samples)})"
        )
    
    def _load_data_for_batch_alternation(self, args, tokenizer, processor):
        """加载数据用于batch-level交替模式"""
        math_path = getattr(args, 'math_data_path', None)
        qa_path = getattr(args, 'qa_data_path', None)

        self.math_samples = self._load_math_data(args, tokenizer, processor, math_path) if math_path else []
        self.qa_samples = self._load_qa_data(args, tokenizer, processor, qa_path) if qa_path else []

        if not self.math_samples and not self.qa_samples:
            raise ValueError("No data loaded! Both math and QA datasets are empty.")

        if args.rollout_shuffle:
            rng = random.Random(args.rollout_seed)
            self.math_samples = sorted(self.math_samples, key=lambda x: rng.random())
            self.qa_samples = sorted(self.qa_samples, key=lambda x: rng.random())

        logger.info(f"Loaded {len(self.math_samples)} math samples, {len(self.qa_samples)} QA samples")
        logger.info(f"Batch alternation mode: {self.math_batches_per_cycle} math batches, "
                    f"{self.qa_batches_per_cycle} QA batches per cycle, "
                    f"start_task={self.batch_alternation_start_task}")

        self._init_batch_alternator()
        if self.count_aware_alternation:
            self.count_aware_cycle_counts = {"math": 0, "qa": 0}
            self.count_aware_train_task = "qa"
            self.count_aware_post_update_task = "math"
            self.count_aware_decision_version = None
            self.version_task_counts = {}
            logger.info(
                "Initialized count-aware alternation: default train=qa/post_update=math, "
                f"math_quota={self.math_batches_per_cycle}, qa_quota={self.qa_batches_per_cycle}"
            )
        self.origin_samples = self.math_samples + self.qa_samples
        self.samples = self.origin_samples

    def _load_data_for_dynamic_alternation(self, args, tokenizer, processor):
        """加载数据用于动态交替模式"""
        math_path = getattr(args, 'math_data_path', None)
        qa_path = getattr(args, 'qa_data_path', None)

        self.math_samples = self._load_math_data(args, tokenizer, processor, math_path) if math_path else []
        self.qa_samples = self._load_qa_data(args, tokenizer, processor, qa_path) if qa_path else []

        if not self.math_samples and not self.qa_samples:
            raise ValueError("No data loaded! Both math and QA datasets are empty.")

        if args.rollout_shuffle:
            rng = random.Random(args.rollout_seed)
            self.math_samples = sorted(self.math_samples, key=lambda x: rng.random())
            self.qa_samples = sorted(self.qa_samples, key=lambda x: rng.random())

        logger.info(f"Loaded {len(self.math_samples)} math samples, {len(self.qa_samples)} QA samples for dynamic alternation")

        self._init_dynamic_alternator()
        self.origin_samples = self.math_samples + self.qa_samples
        self.samples = self.origin_samples

    def _init_dynamic_alternator(self):
        """初始化动态交替调度器（基于版本滞后的连续比例调控）"""
        self.math_offset = 0
        self.qa_offset = 0
        # {version: {"math": count, "qa": count}} — 每个 version 的训练数据组成
        self.version_task_counts = {}
        # {group_index: "math"|"qa"} — 当前 in-flight group 的任务类型
        self.in_flight_groups = {}
        logger.info(f"Initialized dynamic alternator (ratio-based). lag_version: {self.lag_version}")

    def _init_batch_alternator(self):
        """初始化batch交替调度器"""
        # 获取batch size
        self.batch_size = getattr(self.args, 'rollout_batch_size', 32)
        self.samples_per_prompt = getattr(self.args, 'n_samples_per_prompt', 1)
        
        # 计算每个batch需要多少个不同的prompt
        self.prompts_per_batch = self.batch_size
        self.samples_per_batch = self.samples_per_prompt * self.batch_size
        
        # 创建交替序列
        blocks = {
            "math": ["math"] * self.math_batches_per_cycle,
            "qa": ["qa"] * self.qa_batches_per_cycle,
        }
        first = self.batch_alternation_start_task
        second = "qa" if first == "math" else "math"
        self.batch_sequence = blocks[first] + blocks[second]
        
        # 计算总共需要多少个batch
        total_math_prompts = len(self.math_samples)
        total_qa_prompts = len(self.qa_samples)
        
        total_math_batches = (total_math_prompts + self.prompts_per_batch - 1) // self.prompts_per_batch
        total_qa_batches = (total_qa_prompts + self.prompts_per_batch - 1) // self.prompts_per_batch
        total_batches_needed = total_math_batches + total_qa_batches
        
        # 扩展batch_sequence到需要的长度
        cycle_length = len(self.batch_sequence)
        if cycle_length > 0:
            num_cycles = (total_batches_needed + cycle_length - 1) // cycle_length
            self.batch_sequence = self.batch_sequence * num_cycles
            self.batch_sequence = self.batch_sequence[:total_batches_needed]
        else:
            self.batch_sequence = []
        
        # 初始化指针
        self.current_batch_idx = 0
        self.math_offset = 0
        self.qa_offset = 0
        
        logger.info(f"Initialized batch alternator with {len(self.batch_sequence)} batches")
        if self.batch_sequence:
            logger.info(f"Batch sequence preview: {self.batch_sequence[:min(20, len(self.batch_sequence))]}...")
        logger.info(f"Math samples: {total_math_prompts} prompts -> {total_math_batches} batches")
        logger.info(f"QA samples: {total_qa_prompts} prompts -> {total_qa_batches} batches")
    
    def _mix_samples(self, math_samples, qa_samples, terminal_samples=None):
        """按配置比例混合各领域的 Sample 列表。

        普通模式下如果某一类数据较少，用完后从头循环使用，避免 epoch
        尾部退化成单一领域。terminal_ratio=0 时保持原有两领域语义。
        """
        terminal_samples = terminal_samples or []
        if self.terminal_ratio == 0:
            return self._mix_two_domain_samples(math_samples, qa_samples)

        sources = {
            "math": list(math_samples),
            "qa": list(qa_samples),
            "terminal": list(terminal_samples),
        }
        ratios = {
            "math": float(self.math_ratio),
            "qa": float(1.0 - self.math_ratio - self.terminal_ratio),
            "terminal": float(self.terminal_ratio),
        }
        active = [name for name, ratio in ratios.items() if ratio > 0]
        if len(active) == 1:
            return sources[active[0]]

        if self.args.rollout_shuffle:
            rng = random.Random(self.args.rollout_seed)
            for name in active:
                sources[name] = sorted(sources[name], key=lambda x: rng.random())

        for name in active:
            if not sources[name]:
                raise ValueError(f"{name} ratio is positive but its dataset is empty")

        total_len = max(math.ceil(len(sources[name]) / ratios[name]) for name in active)
        while True:
            targets = {name: int(total_len * ratios[name]) for name in active}
            remainder = total_len - sum(targets.values())
            order = sorted(
                active,
                key=lambda name: total_len * ratios[name] - targets[name],
                reverse=True,
            )
            for name in order[:remainder]:
                targets[name] += 1
            if all(targets[name] >= len(sources[name]) for name in active):
                break
            total_len += 1

        task_sequence = [
            task_type
            for task_type in active
            for _ in range(targets[task_type])
        ]
        rng_mix = random.Random(self.args.rollout_seed)
        rng_mix.shuffle(task_sequence)

        mixed = []
        offsets = {name: 0 for name in active}
        for task_type in task_sequence:
            source = sources[task_type]
            mixed.append(source[offsets[task_type] % len(source)])
            offsets[task_type] += 1

        logger.info(
            "Mixed normal data with recycling: total=%d, targets=%s",
            len(mixed),
            targets,
        )

        return mixed

    def _mix_two_domain_samples(self, math_samples, qa_samples):
        """Preserve the pre-Terminal Math/QA mixing behavior exactly."""
        if not math_samples:
            return qa_samples
        if not qa_samples:
            return math_samples

        if self.args.rollout_shuffle:
            rng = random.Random(self.args.rollout_seed)
            math_samples = sorted(math_samples, key=lambda sample: rng.random())
            qa_samples = sorted(qa_samples, key=lambda sample: rng.random())

        math_len, qa_len = len(math_samples), len(qa_samples)
        math_ratio = max(0.0, min(1.0, self.math_ratio))
        if math_ratio == 1.0:
            return math_samples
        if math_ratio == 0.0:
            return qa_samples

        total_len = max(
            math.ceil(math_len / math_ratio),
            math.ceil(qa_len / (1.0 - math_ratio)),
        )
        while True:
            target_math = round(total_len * math_ratio)
            target_qa = total_len - target_math
            if target_math >= math_len and target_qa >= qa_len:
                break
            total_len += 1

        task_sequence = ["math"] * target_math + ["qa"] * target_qa
        random.Random(self.args.rollout_seed).shuffle(task_sequence)
        mixed = []
        math_ptr = qa_ptr = 0
        for task_type in task_sequence:
            if task_type == "math":
                mixed.append(math_samples[math_ptr % math_len])
                math_ptr += 1
            else:
                mixed.append(qa_samples[qa_ptr % qa_len])
                qa_ptr += 1
        return mixed
    
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """获取指定数量的样本用于rollout
        
        支持两种模式：
        1. 普通模式：完全随机混合
        2. Batch-level交替模式：按batch粒度交替使用不同类型的任务
        
        Args:
            num_samples: 应该是batch_size（即需要多少个不同的prompt）
        
        Returns:
            list of groups，每个 group 包含 n_samples_per_prompt 个相同 prompt 的样本
        """
        if not self.origin_samples and not self.buffer:
            return []

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)
        if num_samples == 0:
            return samples

        # 如果使用特定的交替模式
        if self.dynamic_alternation:
            samples += self._get_samples_dynamic(num_samples, selected_buffer_groups=samples)
        elif self.count_aware_alternation:
            samples += self._get_samples_count_aware(num_samples)
        elif self.batch_alternation:
            samples += self._get_samples_alternating(num_samples)
        else:
            samples += self._get_samples_normal(num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        num_to_pop = min(len(self.buffer), num_samples)
        samples = self.buffer[:num_to_pop]
        del self.buffer[:num_to_pop]

        if self.dynamic_alternation:
            for group in samples:
                self._register_in_flight_group_from_existing_group(group)

        return samples

    def _register_in_flight_group_from_existing_group(self, group: list[Sample]):
        """Track recycled/partial groups so dynamic alternation sees their stale pressure."""
        if not group:
            return
        group_index = getattr(group[0], "group_index", None)
        if group_index is None:
            return
        task_type, dispatch_version = self._get_group_task_and_dispatch_version(group)
        if task_type not in ("math", "qa") or dispatch_version is None:
            return
        self.in_flight_groups[group_index] = (task_type, dispatch_version)

    def _get_lag_sample_for_group(self, task_type: str, dispatch_version: int, current_version: int) -> int:
        """Use the same lag_sample definition as the off-policy mask."""
        if dispatch_version is None or dispatch_version >= current_version:
            return 0

        lag_sample = 0
        for version in range(dispatch_version, current_version):
            counts = self.version_task_counts.get(version, {})
            lag_sample += counts.get(task_type, 0)
        return lag_sample

    def _get_group_task_and_dispatch_version(self, group: list[Sample]):
        if not group:
            return None, None
        metadata = group[0].metadata if isinstance(group[0].metadata, dict) else {}
        task_type = metadata.get("task_type", "math")
        dispatch_version = metadata.get("dispatch_version", None)
        return task_type, dispatch_version

    def _accumulate_lag_metrics_for_group(self, metrics: dict, group: list[Sample], current_version: int, source: str):
        task_type, dispatch_version = self._get_group_task_and_dispatch_version(group)
        if task_type not in ("math", "qa"):
            return

        version_lag = max(0, current_version - dispatch_version) if dispatch_version is not None else 0
        group_lag_sample = self._get_lag_sample_for_group(task_type, dispatch_version, current_version)

        metrics[f"{source}_{task_type}"] += 1
        metrics[f"in_flight_{task_type}"] += 1
        metrics[f"max_lag_{task_type}"] = max(metrics[f"max_lag_{task_type}"], version_lag)
        metrics[f"lag_sample_{task_type}"] += group_lag_sample

    def _get_samples_dynamic(self, num_samples: int, selected_buffer_groups=None) -> list[list[Sample]]:
        """Dynamic交替模式的样本获取（基于版本滞后的连续比例调控）

        每次派发 num_samples 个 group 时：
        1. 清理已完成的 in-flight group
        2. 对所有 in-flight group 累计各自的 lag_sample
        3. 按反比概率决定每个 group 的任务类型
        4. 用 math_ratio 做平滑，并用 min/max math ratio 限制波动
        """
        num_prompts = num_samples
        current_version = getattr(self.args, 'current_policy_version', 0)

        # ── 1. 清理已完成的 in-flight group ──
        import fully_async_rollout as rollout_mod
        worker = rollout_mod.get_existing_worker()
        if worker is not None:
            active_groups = worker._current_stale_sample_ids()
            for gid in list(self.in_flight_groups.keys()):
                if gid not in active_groups:
                    del self.in_flight_groups[gid]

        # ── 2. 累计 active in-flight、已选 recycled buffer、剩余 buffer 的 lag_sample ──
        metrics = {
            "lag_sample_math": 0,
            "lag_sample_qa": 0,
            "max_lag_math": 0,
            "max_lag_qa": 0,
            "in_flight_math": 0,
            "in_flight_qa": 0,
            "active_math": 0,
            "active_qa": 0,
            "selected_buffer_math": 0,
            "selected_buffer_qa": 0,
            "pending_buffer_math": 0,
            "pending_buffer_qa": 0,
        }

        for task_type, dispatch_version in self.in_flight_groups.values():
            version_lag = max(0, current_version - dispatch_version)
            group_lag_sample = self._get_lag_sample_for_group(task_type, dispatch_version, current_version)
            if task_type == "math":
                metrics["active_math"] += 1
                metrics["in_flight_math"] += 1
                metrics["max_lag_math"] = max(metrics["max_lag_math"], version_lag)
                metrics["lag_sample_math"] += group_lag_sample
            elif task_type == "qa":
                metrics["active_qa"] += 1
                metrics["in_flight_qa"] += 1
                metrics["max_lag_qa"] = max(metrics["max_lag_qa"], version_lag)
                metrics["lag_sample_qa"] += group_lag_sample

        # selected_buffer_groups were already registered into in_flight_groups by
        # _get_samples_from_buffer(). Count remaining buffer groups separately.
        for group in self.buffer:
            self._accumulate_lag_metrics_for_group(metrics, group, current_version, "pending_buffer")

        lag_sample_math = metrics["lag_sample_math"]
        lag_sample_qa = metrics["lag_sample_qa"]
        max_lag_math = metrics["max_lag_math"]
        max_lag_qa = metrics["max_lag_qa"]
        in_flight_math = metrics["in_flight_math"]
        in_flight_qa = metrics["in_flight_qa"]

        # ── 3. 计算派发比例 ──
        base_math_prob = getattr(self.args, 'math_ratio', 0.5)
        warmup_steps = max(0, getattr(self.args, 'dynamic_alternation_warmup_steps', 5))
        in_warmup = current_version < warmup_steps
        if in_warmup:
            lag_based_math_prob = base_math_prob
            alpha = 0.0
            math_prob = base_math_prob
        else:
            if lag_sample_math == 0 and lag_sample_qa == 0:
                lag_based_math_prob = base_math_prob
            else:
                total = lag_sample_math + lag_sample_qa
                lag_based_math_prob = lag_sample_qa / total  # 反比：QA 更 stale 时少派 QA，多派 math

            alpha = max(0.0, min(1.0, getattr(self.args, 'dynamic_alternation_alpha', 0.5)))
            min_math_prob = getattr(self.args, 'dynamic_alternation_min_math_ratio', 0.3)
            max_math_prob = getattr(self.args, 'dynamic_alternation_max_math_ratio', 0.7)
            if min_math_prob > max_math_prob:
                min_math_prob, max_math_prob = max_math_prob, min_math_prob

            smoothed_math_prob = (1.0 - alpha) * base_math_prob + alpha * lag_based_math_prob
            math_prob = max(min_math_prob, min(max_math_prob, smoothed_math_prob))

        logger.info(
            f"[v{current_version}] dynamic_alternation: "
            f"in_flight_math={in_flight_math}, in_flight_qa={in_flight_qa}, "
            f"active_math={metrics['active_math']}, active_qa={metrics['active_qa']}, "
            f"selected_buffer_math={metrics['selected_buffer_math']}, selected_buffer_qa={metrics['selected_buffer_qa']}, "
            f"pending_buffer_math={metrics['pending_buffer_math']}, pending_buffer_qa={metrics['pending_buffer_qa']}, "
            f"max_lag_math={max_lag_math}, max_lag_qa={max_lag_qa}, "
            f"lag_sample_math={lag_sample_math}, lag_sample_qa={lag_sample_qa}, "
            f"base_math_prob={base_math_prob:.3f}, lag_based_math_prob={lag_based_math_prob:.3f}, "
            f"alpha={alpha:.3f}, warmup_steps={warmup_steps}, in_warmup={in_warmup}, "
            f"math_prob={math_prob:.3f}, in_flight={len(self.in_flight_groups)}"
        )

        # ── 4. 按概率逐个派发 group ──
        rng = random.Random(self.args.rollout_seed + self.sample_group_index)
        samples = []
        dispatched_math = 0
        dispatched_qa = 0

        for _ in range(num_prompts):
            # 决定这个 group 的任务类型
            if rng.random() < math_prob:
                task_type = "math"
            else:
                task_type = "qa"

            # 取对应数据源的样本
            if task_type == "math":
                source_samples = self.math_samples
                offset = self.math_offset
            else:
                source_samples = self.qa_samples
                offset = self.qa_offset

            if not source_samples:
                # 该类数据为空，fallback 到另一类
                task_type = "qa" if task_type == "math" else "math"
                source_samples = self.qa_samples if task_type == "qa" else self.math_samples
                offset = self.qa_offset if task_type == "qa" else self.math_offset

            if not source_samples:
                continue  # 两类都没数据

            # 取一个 prompt
            if offset >= len(source_samples):
                # 数据耗尽，shuffle 重新开始
                if self.args.rollout_shuffle:
                    rng_shuffle = random.Random(self.args.rollout_seed + self.epoch_id)
                    source_samples = sorted(source_samples, key=lambda x: rng_shuffle.random())
                    if task_type == "math":
                        self.math_samples = source_samples
                    else:
                        self.qa_samples = source_samples
                    self.epoch_id += 1
                offset = 0

            prompt_sample = source_samples[offset]
            new_offset = (offset + 1) % len(source_samples)

            # 更新 offset
            if task_type == "math":
                self.math_offset = new_offset
            else:
                self.qa_offset = new_offset

            # 构建 group
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                # 记录 dispatch version，用于 per-group off-policy mask 判断
                if not isinstance(sample.metadata, dict):
                    sample.metadata = {}
                sample.metadata["dispatch_version"] = current_version
                self.sample_index += 1
                group.append(sample)

            # 记录 in-flight (group_index → (task_type, version))
            self.in_flight_groups[self.sample_group_index] = (task_type, current_version)
            # version_task_counts 由 trainer 侧在 RolloutManager.generate() 中记录

            if task_type == "math":
                dispatched_math += 1
            else:
                dispatched_qa += 1

            self.sample_group_index += 1
            samples.append(group)

        logger.info(
            f"[v{current_version}] dispatched: math={dispatched_math}, qa={dispatched_qa}, total={len(samples)}"
        )

        return samples

    def _count_aware_cycle_complete(self) -> bool:
        return (
            self.count_aware_cycle_counts["math"] >= max(0, int(self.math_batches_per_cycle))
            and self.count_aware_cycle_counts["qa"] >= max(0, int(self.qa_batches_per_cycle))
        )

    def _update_count_aware_direction(self):
        """Choose the next complete cycle from the latest trained composition."""
        if not self.version_task_counts:
            return

        decision_version = max(self.version_task_counts)
        if decision_version == self.count_aware_decision_version:
            return

        counts = self.version_task_counts[decision_version]
        math_count = int(counts.get("math", 0))
        qa_count = int(counts.get("qa", 0))
        if math_count > qa_count:
            # The collected batch is math-heavy: use adaptive-math next.
            self.count_aware_train_task = "qa"
            self.count_aware_post_update_task = "math"
        elif qa_count > math_count:
            # The collected batch is QA-heavy: use adaptive-QA next.
            self.count_aware_train_task = "math"
            self.count_aware_post_update_task = "qa"
        # A tie deliberately keeps the previous direction.
        self.count_aware_decision_version = decision_version
        logger.info(
            f"[v{decision_version}] count-aware decision: trained_math={math_count}, trained_qa={qa_count}, "
            f"train_task={self.count_aware_train_task}, "
            f"post_update_task={self.count_aware_post_update_task}"
        )

    def _choose_count_aware_task(self) -> str:
        if self._count_aware_cycle_complete():
            self.count_aware_cycle_counts = {"math": 0, "qa": 0}
            self._update_count_aware_direction()

        phase = getattr(self.args, "current_policy_phase", "post_update")
        preferred = self.count_aware_train_task if phase == "train" else self.count_aware_post_update_task
        fallback = "qa" if preferred == "math" else "math"
        quotas = {
            "math": max(0, int(self.math_batches_per_cycle)),
            "qa": max(0, int(self.qa_batches_per_cycle)),
        }
        for task_type in (preferred, fallback):
            if self.count_aware_cycle_counts[task_type] < quotas[task_type]:
                self.count_aware_cycle_counts[task_type] += 1
                return task_type
        return "math" if random.random() < self.math_ratio else "qa"

    def _get_one_count_aware_prompt(self, task_type: str):
        source_samples = self.math_samples if task_type == "math" else self.qa_samples
        offset = self.math_offset if task_type == "math" else self.qa_offset
        if not source_samples:
            return None
        if offset >= len(source_samples):
            offset = 0
        prompt_sample = source_samples[offset]
        offset = (offset + 1) % len(source_samples)
        if task_type == "math":
            self.math_offset = offset
        else:
            self.qa_offset = offset
        return prompt_sample

    def _get_samples_count_aware(self, num_samples: int) -> list[list[Sample]]:
        current_version = getattr(self.args, "current_policy_version", 0)
        phase = getattr(self.args, "current_policy_phase", "post_update")
        samples = []
        dispatched = {"math": 0, "qa": 0}
        for _ in range(num_samples):
            task_type = self._choose_count_aware_task()
            prompt_sample = self._get_one_count_aware_prompt(task_type)
            if prompt_sample is None:
                task_type = "qa" if task_type == "math" else "math"
                prompt_sample = self._get_one_count_aware_prompt(task_type)
            if prompt_sample is None:
                continue

            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                if not isinstance(sample.metadata, dict):
                    sample.metadata = {}
                sample.metadata["dispatch_version"] = current_version
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
            dispatched[task_type] += 1

        logger.info(
            f"[v{current_version}] count-aware dispatch: phase={phase}, "
            f"train_task={self.count_aware_train_task}, post_update_task={self.count_aware_post_update_task}, "
            f"cycle_counts={self.count_aware_cycle_counts}, dispatched={dispatched}"
        )
        return samples

    def _get_samples_alternating(self, num_samples: int) -> list[list[Sample]]:
        """Batch-level交替模式的样本获取"""
        # num_samples 就是 batch_size，需要多少个不同的prompt
        num_prompts = num_samples
        
        # 检查是否需要进入下一个epoch
        if self.current_batch_idx >= len(self.batch_sequence):
            # 所有batch都用完了，进入新epoch
            self.epoch_id += 1
            if self.args.rollout_shuffle:
                self._reshuffle_for_new_epoch()
            self.current_batch_idx = 0
            self.math_offset = 0
            self.qa_offset = 0
        
        # 获取当前batch类型
        batch_type = self.batch_sequence[self.current_batch_idx]
        self.current_batch_idx += 1
        
        # 根据batch类型选择数据源
        if batch_type == "math":
            source_samples = self.math_samples
            offset = self.math_offset
        else:  # "qa"
            source_samples = self.qa_samples
            offset = self.qa_offset
        
        # 获取需要的prompts
        if offset + num_prompts <= len(source_samples):
            prompt_samples = source_samples[offset:offset + num_prompts]
            if batch_type == "math":
                self.math_offset += num_prompts
            else:
                self.qa_offset += num_prompts
        else:
            # 当前类型数据不够，循环使用
            prompt_samples = source_samples[offset:]
            remaining_prompts = num_prompts - len(prompt_samples)
            
            # 从头开始取剩余部分
            prompt_samples.extend(source_samples[:remaining_prompts])
            
            if batch_type == "math":
                self.math_offset = remaining_prompts
            else:
                self.qa_offset = remaining_prompts
            
            # 记录循环使用的情况
            logger.info(f"Recycling {batch_type} data: used {len(prompt_samples)} prompts, "
                       f"new offset={self.math_offset if batch_type=='math' else self.qa_offset}")
        
        # 为每个prompt生成 n_samples_per_prompt 个样本（复制）
        current_version = getattr(self.args, 'current_policy_version', 0)
        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                if not isinstance(sample.metadata, dict):
                    sample.metadata = {}
                sample.metadata["dispatch_version"] = current_version
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)

        # 记录batch信息（可选，用于调试）
        if hasattr(self.args, 'debug') and self.args.debug:
            logger.debug(f"Batch {self.current_batch_idx-1}: type={batch_type}, "
                        f"prompts={len(prompt_samples)}, samples={len(samples)}")
        
        return samples
    
    def _get_samples_normal(self, num_samples: int) -> list[list[Sample]]:
        """普通模式的样本获取（原有逻辑）"""
        # 获取 num_samples 个 prompts
        num_prompts = num_samples
        
        if self.sample_offset + num_prompts <= len(self.samples):
            prompt_samples = self.samples[self.sample_offset : self.sample_offset + num_prompts]
            self.sample_offset += num_prompts
        else:
            # 数据耗尽，需要跨 epoch
            prompt_samples = self.samples[self.sample_offset :]
            remaining_prompts = num_prompts - len(prompt_samples)
            
            self.epoch_id += 1
            if self.args.rollout_shuffle:
                self.shuffle(self.epoch_id)
            
            self.sample_offset = 0
            if remaining_prompts > 0:
                prompt_samples.extend(self.samples[:remaining_prompts])
                self.sample_offset = remaining_prompts
        
        # 为每个 prompt 生成 n_samples_per_prompt 个样本（复制）
        current_version = getattr(self.args, 'current_policy_version', 0)
        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                if not isinstance(sample.metadata, dict):
                    sample.metadata = {}
                sample.metadata["dispatch_version"] = current_version
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)

        return samples
    
    def add_samples(self, samples: list[list[Sample]]):
        """将样本组回灌到缓冲区，优先用于后续rollout。"""
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"

        for group in samples:
            assert (
                len(group) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(group)} != {self.args.n_samples_per_prompt}"
            # 清理 in-flight tracking（回灌的 group 不再是 in-flight）
            if hasattr(self, 'in_flight_groups') and group:
                gid = getattr(group[0], 'group_index', None)
                if gid is not None:
                    self.in_flight_groups.pop(gid, None)
            self.buffer.append(group)
    
    def save(self, rollout_id):
        """保存数据源状态（checkpoint）"""
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        
        # 如果是batch alternation模式，额外保存状态
        if self.batch_alternation or self.count_aware_alternation:
            state_dict.update({
                "batch_alternation": True,
                "current_batch_idx": self.current_batch_idx,
                "math_offset": self.math_offset,
                "qa_offset": self.qa_offset,
                "batch_sequence": self.batch_sequence,
            })
        if self.count_aware_alternation:
            state_dict.update({
                "count_aware_alternation": True,
                "count_aware_cycle_counts": self.count_aware_cycle_counts,
                "count_aware_train_task": self.count_aware_train_task,
                "count_aware_post_update_task": self.count_aware_post_update_task,
                "count_aware_decision_version": self.count_aware_decision_version,
                "version_task_counts": self.version_task_counts,
            })

        # 如果是dynamic alternation模式，额外保存状态
        if self.dynamic_alternation:
            state_dict.update({
                "dynamic_alternation": True,
                "math_offset": self.math_offset,
                "qa_offset": self.qa_offset,
                "version_task_counts": self.version_task_counts,
            })
        
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        """加载数据源状态（从checkpoint）"""
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load metadata from {path}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})
        
        # 如果是batch alternation模式，恢复相关状态
        if (self.batch_alternation or self.count_aware_alternation) and state_dict.get("batch_alternation", False):
            self.current_batch_idx = state_dict.get("current_batch_idx", 0)
            self.math_offset = state_dict.get("math_offset", 0)
            self.qa_offset = state_dict.get("qa_offset", 0)
            # batch_sequence不需要从checkpoint恢复，因为它是根据配置生成的
            # 但如果配置可能变化，可以恢复并验证一致性
            saved_sequence = state_dict.get("batch_sequence", [])
            if saved_sequence and saved_sequence != self.batch_sequence:
                logger.warning(f"Saved batch sequence differs from current config. "
                             f"Using current config but offset may be mismatched.")
        if self.count_aware_alternation and state_dict.get("count_aware_alternation", False):
            self.count_aware_cycle_counts = state_dict.get("count_aware_cycle_counts", {"math": 0, "qa": 0})
            self.count_aware_train_task = state_dict.get("count_aware_train_task", "qa")
            self.count_aware_post_update_task = state_dict.get("count_aware_post_update_task", "math")
            self.count_aware_decision_version = state_dict.get("count_aware_decision_version")
            self.version_task_counts = state_dict.get("version_task_counts", {})

        # 如果是dynamic alternation模式，恢复相关状态
        if self.dynamic_alternation and state_dict.get("dynamic_alternation", False):
            self.math_offset = state_dict.get("math_offset", 0)
            self.qa_offset = state_dict.get("qa_offset", 0)
            self.version_task_counts = state_dict.get("version_task_counts", {})
            # in_flight_groups 不需要恢复（resume 时 worker 无 active group，会被自然清理）
            logger.info(f"Restored dynamic alternation state: math_offset={self.math_offset}, "
                       f"qa_offset={self.qa_offset}, version_history={len(self.version_task_counts)} versions")
        
        # 保证断点续训时，当前 epoch 的数据打乱顺序与中断前完全一致
        if self.args.rollout_global_dataset and self.args.rollout_shuffle and not self.batch_alternation and not self.count_aware_alternation:
            self.shuffle(self.epoch_id)
    
    def shuffle(self, new_epoch_id):
        """为新的epoch打乱样本（仅用于普通模式）"""
        if self.batch_alternation or self.count_aware_alternation:
            logger.warning("shuffle() called in batch alternation mode, but should use _reshuffle_for_new_epoch()")
            return
            
        if self.epoch_id == new_epoch_id and len(self.samples) == len(self.origin_samples):
            # 防止在初始化后或load后重复shuffle同个epoch
            return
            
        rng = random.Random(self.args.rollout_seed + new_epoch_id)
        permutation = list(range(len(self.origin_samples)))
        rng.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id
        # reset offset for new epoch
        self.sample_offset = 0
    
    def _reshuffle_for_new_epoch(self):
        """新epoch时重新组织batch顺序和打乱数据（仅用于batch alternation模式）"""
        # 重新初始化batch序列（确保配置没变）
        self._init_batch_alternator()
        
        # 如果配置了shuffle，分别打乱math和QA数据
        if self.args.rollout_shuffle:
            rng = random.Random(self.args.rollout_seed + self.epoch_id)
            
            # 打乱math数据
            math_indices = list(range(len(self.math_samples)))
            rng.shuffle(math_indices)
            self.math_samples = [self.math_samples[i] for i in math_indices]
            
            # 打乱QA数据
            qa_indices = list(range(len(self.qa_samples)))
            rng.shuffle(qa_indices)
            self.qa_samples = [self.qa_samples[i] for i in qa_indices]
            
            logger.info(f"Shuffled math and QA data for epoch {self.epoch_id}")
        
        # 重置指针
        self.current_batch_idx = 0
        self.math_offset = 0
        self.qa_offset = 0
        
        logger.info(f"Reset batch alternator for epoch {self.epoch_id}")
