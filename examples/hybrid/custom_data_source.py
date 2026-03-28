"""Custom DataSource for unified math and QA data loading in slime framework"""

import json
import logging
import pandas as pd
import random
import copy
import os
import torch
from typing import Dict, List, Any
from pathlib import Path

from slime.rollout.data_source import DataSource
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class CustomDataSource(DataSource):
    """统一的数据源，支持math和qa两种数据格式的混合加载"""
    
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
            - qa_data_path: QA数据路径  
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
        
        # 获取batch-level交替参数
        self.batch_alternation = getattr(args, 'batch_alternation', False)
        self.math_batches_per_cycle = getattr(args, 'math_batches_per_cycle', 1)
        self.qa_batches_per_cycle = getattr(args, 'qa_batches_per_cycle', 1)
        
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
            if self.batch_alternation:
                self._load_data_for_batch_alternation(args, tokenizer, processor)
            else:
                self._load_unified_data(args, tokenizer, processor)
            
            if self.args.rollout_shuffle and not self.batch_alternation:
                # 只在普通模式下shuffle，batch-level模式有自己的shuffle逻辑
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
    
    def _load_math_data(self, math_path: str) -> List[Dict]:
        """加载数学数据"""
        data = []
        if not math_path:
            return data
            
        try:
            with open(math_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    data.append({
                        "prompt": item["prompt"],
                        "label": str(item["label"]),
                        "task_type": "math",
                        "tools_available": ["code_interpreter"],
                        "metadata": {
                            "raw_label": item.get("label"), 
                            "task_type": "math",
                            "tools_available": ["code_interpreter"]
                        }
                    })
            logger.info(f"Loaded {len(data)} math samples from {math_path}")
        except Exception as e:
            logger.error(f"Failed to load math data from {math_path}: {e}")
            raise ValueError(f"Failed to load math data: {e}")
        
        return data
    
    def _load_qa_data(self, qa_path: str) -> List[Dict]:
        """加载QA数据"""
        data = []
        if not qa_path:
            return data
            
        try:
            df = pd.read_parquet(qa_path)
            
            for _, row in df.iterrows():
                # 从reward_model提取ground_truth
                reward_model = row.get("reward_model", {})
                if isinstance(reward_model, dict) and "ground_truth" in reward_model:
                    ground_truth = reward_model["ground_truth"]
                else:
                    ground_truth = str(row.get("golden_answers", ""))
                
                data.append({
                    "prompt": row["prompt"],
                    "label": ground_truth,
                    "task_type": "qa",
                    "tools_available": ["search"],
                    "metadata": {
                        "id": row.get("id"),
                        "question": row.get("question"),
                        "data_source": row.get("data_source"),
                        "ability": row.get("ability"),
                        "golden_answers": row.get("golden_answers"),
                        "task_type": "qa",
                        "tools_available": ["search"]
                    }
                })
            logger.info(f"Loaded {len(data)} QA samples from {qa_path}")
        except Exception as e:
            logger.error(f"Failed to load QA data from {qa_path}: {e}")
            raise ValueError(f"Failed to load QA data: {e}")
        
        return data
    
    def _convert_to_samples(self, data: List[Dict], tokenizer, processor, task_type: str) -> List[Sample]:
        """将原始数据转换为Sample对象列表"""
        samples = []
        for idx, item in enumerate(data):
            prompt = item["prompt"]
            label = item["label"]
            metadata = item.get("metadata", {})
            
            # 应用聊天模板
            if self.args.apply_chat_template:
                if isinstance(prompt, str):
                    prompt = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(self.args.apply_chat_template_kwargs or {})
                )
            else:
                formatted_prompt = prompt if isinstance(prompt, str) else str(prompt)
            
            # 创建Sample对象
            sample_kwargs = {
                "prompt": formatted_prompt,
                "label": label,
                "metadata": metadata,
                "index": idx
            }
            
            if hasattr(Sample, 'tools'):
                sample_kwargs["tools"] = item.get("tools_available", [])
            
            sample = Sample(**sample_kwargs)
            samples.append(sample)
        
        logger.info(f"Converted {len(samples)} samples for task type: {task_type}")
        return samples
    
    def _load_unified_data(self, args, tokenizer, processor):
        """加载并混合math和qa数据（普通模式）"""
        # 获取数据路径和混合比例
        math_path = getattr(args, 'math_data_path', None)
        qa_path = getattr(args, 'qa_data_path', None)
        math_ratio = getattr(args, 'math_ratio', 0.7)
        
        # 根据比例灵活加载数据
        if math_ratio == 1.0:
            if not math_path:
                raise ValueError("math_data_path is required when math_ratio=1.0")
            logger.info("math_ratio=1.0: loading only math data")
            math_data = self._load_math_data(math_path)
            qa_data = []
        elif math_ratio == 0.0:
            if not qa_path:
                raise ValueError("qa_data_path is required when math_ratio=0.0")
            logger.info("math_ratio=0.0: loading only QA data")
            math_data = []
            qa_data = self._load_qa_data(qa_path)
        else:
            if not math_path or not qa_path:
                raise ValueError(
                    f"Both math_data_path and qa_data_path are required when 0 < math_ratio < 1. "
                    f"Got math_ratio={math_ratio}, math_path={math_path}, qa_path={qa_path}"
                )
            logger.info(f"Loading mixed data with math_ratio={math_ratio}")
            math_data = self._load_math_data(math_path)
            qa_data = self._load_qa_data(qa_path)
        
        if not math_data and not qa_data:
            raise ValueError("No data loaded! Both math and QA datasets are empty.")
        
        logger.info(f"Loaded {len(math_data)} math samples, {len(qa_data)} QA samples")
        
        # 按比例混合数据
        all_raw_data = self._mix_data(math_data, qa_data)
        
        # 转换为Sample对象
        for item in all_raw_data:
            prompt = item["prompt"]
            label = item["label"]
            metadata = item.get("metadata", {})
            
            # 应用聊天模板
            if args.apply_chat_template:
                if isinstance(prompt, str):
                    prompt = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(args.apply_chat_template_kwargs or {})
                )
            else:
                formatted_prompt = prompt if isinstance(prompt, str) else str(prompt)
            
            # 创建Sample对象
            sample_kwargs = {
                "prompt": formatted_prompt,
                "label": label,
                "metadata": metadata,
                "index": len(self.origin_samples)
            }
            if hasattr(Sample, 'tools'):
                sample_kwargs["tools"] = item.get("tools_available", [])
            
            sample = Sample(**sample_kwargs)
            self.origin_samples.append(sample)
        
        self.samples = self.origin_samples
        logger.info(
            f"Loaded {len(self.origin_samples)} total samples "
            f"(math: {len(math_data)}, qa: {len(qa_data)})"
        )
    
    def _load_data_for_batch_alternation(self, args, tokenizer, processor):
        """加载数据用于batch-level交替模式"""
        math_path = getattr(args, 'math_data_path', None)
        qa_path = getattr(args, 'qa_data_path', None)
        
        # 加载原始数据
        math_data = self._load_math_data(math_path) if math_path else []
        qa_data = self._load_qa_data(qa_path) if qa_path else []
        
        if not math_data and not qa_data:
            raise ValueError("No data loaded! Both math and QA datasets are empty.")
        
        logger.info(f"Loaded {len(math_data)} math samples, {len(qa_data)} QA samples")
        logger.info(f"Batch alternation mode: {self.math_batches_per_cycle} math batches, "
                    f"{self.qa_batches_per_cycle} QA batches per cycle")
        
        # 分别转换为Sample对象（不混合）
        self.math_samples = self._convert_to_samples(math_data, tokenizer, processor, "math")
        self.qa_samples = self._convert_to_samples(qa_data, tokenizer, processor, "qa")
        
        # 初始化交替调度器
        self._init_batch_alternator()
        
        # 为了兼容性，origin_samples保持为所有样本的列表
        self.origin_samples = self.math_samples + self.qa_samples
        self.samples = self.origin_samples  # 普通模式会用到，这里保持兼容
    
    def _init_batch_alternator(self):
        """初始化batch交替调度器"""
        # 获取batch size
        self.batch_size = getattr(self.args, 'rollout_batch_size', 32)
        self.samples_per_prompt = getattr(self.args, 'n_samples_per_prompt', 1)
        
        # 计算每个batch需要多少个不同的prompt
        self.prompts_per_batch = self.batch_size
        self.samples_per_batch = self.samples_per_prompt * self.batch_size
        
        # 创建交替序列
        self.batch_sequence = []
        for _ in range(self.math_batches_per_cycle):
            self.batch_sequence.append("math")
        for _ in range(self.qa_batches_per_cycle):
            self.batch_sequence.append("qa")
        
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
    
    def _mix_data(self, math_data: List[Dict], qa_data: List[Dict]) -> List[Dict]:
        """按比例混合math和qa数据"""
        if not math_data:
            return qa_data
        if not qa_data:
            return math_data
        
        # 如果启用shuffle，先分别打乱两个列表
        if self.args.rollout_shuffle:
            rng = random.Random(self.args.rollout_seed)
            math_data = sorted(math_data, key=lambda x: rng.random())
            qa_data = sorted(qa_data, key=lambda x: rng.random())
        
        mixed = []
        math_ptr, qa_ptr = 0, 0
        math_len, qa_len = len(math_data), len(qa_data)
        
        # 根据比例交替选择
        rng_mix = random.Random(self.args.rollout_seed)
        while math_ptr < math_len and qa_ptr < qa_len:
            if rng_mix.random() < self.math_ratio:
                mixed.append(math_data[math_ptr])
                math_ptr += 1
            else:
                mixed.append(qa_data[qa_ptr])
                qa_ptr += 1
        
        # 处理剩余数据
        mixed.extend(math_data[math_ptr:])
        mixed.extend(qa_data[qa_ptr:])
        
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

        # 如果使用batch-level交替模式
        if self.batch_alternation:
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
        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
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
        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
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
        if self.batch_alternation:
            state_dict.update({
                "batch_alternation": True,
                "current_batch_idx": self.current_batch_idx,
                "math_offset": self.math_offset,
                "qa_offset": self.qa_offset,
                "batch_sequence": self.batch_sequence,
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
        if self.batch_alternation and state_dict.get("batch_alternation", False):
            self.current_batch_idx = state_dict.get("current_batch_idx", 0)
            self.math_offset = state_dict.get("math_offset", 0)
            self.qa_offset = state_dict.get("qa_offset", 0)
            # batch_sequence不需要从checkpoint恢复，因为它是根据配置生成的
            # 但如果配置可能变化，可以恢复并验证一致性
            saved_sequence = state_dict.get("batch_sequence", [])
            if saved_sequence and saved_sequence != self.batch_sequence:
                logger.warning(f"Saved batch sequence differs from current config. "
                             f"Using current config but offset may be mismatched.")
        
        # 保证断点续训时，当前 epoch 的数据打乱顺序与中断前完全一致
        if self.args.rollout_global_dataset and self.args.rollout_shuffle and not self.batch_alternation:
            self.shuffle(self.epoch_id)
    
    def shuffle(self, new_epoch_id):
        """为新的epoch打乱样本（仅用于普通模式）"""
        if self.batch_alternation:
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