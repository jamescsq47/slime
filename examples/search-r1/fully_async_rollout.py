import asyncio
import atexit
import queue
import threading
import time
import wandb

# Import core functions from sglang_rollout directly to avoid code duplication
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.types import Sample

# Global worker manager
_global_worker = None
_worker_lock = threading.Lock()
_wandb_metric_defined = False

def get_global_worker(args, data_buffer):
    """Get or create global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=args.sglang_server_concurrency)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    """
    Simplified asynchronous rollout worker, using threads instead of processes
    Supports continuous running, independent of rollout function lifecycle
    """

    def __init__(self, args, data_buffer, concurrency=10):
        self.args = args
        self.data_buffer = data_buffer  # Directly save data_buffer reference
        self.concurrency = concurrency
        self.running = True
        self.output_queue = queue.Queue(maxsize=1000)  # Continuous output queue
        self.worker_thread = None
        self.state = GenerateState(args)
        self.max_queue_size = self.output_queue.maxsize
        self.step_counter = 0  

    async def continuous_worker_loop(self):
        """Continuous work loop - constantly get data from data_buffer and process"""
        print("Continuous async rollout worker started")

        active_tasks = set()
        max_concurrent_tasks = self.args.rollout_batch_size #32
        group_id_counter = 0

        while self.running:
            try:
                # Clean up completed tasks
                if active_tasks:
                    done_tasks = {task for task in active_tasks if task.done()}
                    for task in done_tasks:
                        try:
                            task.result()  # Results are already handled in callbacks
                        except Exception as e:
                            print(f"Task failed with exception: {e}")
                    active_tasks -= done_tasks

                # If active task count hasn't reached limit, try to get new data and start tasks
                while len(active_tasks) < max_concurrent_tasks and self.running:
                    samples = self.data_buffer.get_samples(1)

                    for group in samples:
                        group_id = group_id_counter
                        group_id_counter += 1

                        # Create new async task
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )

                        # Add completion callback
                        def make_callback(gid):
                            def task_done_callback(done_task):
                                result = done_task.result()
                                self.output_queue.put((gid, result))

                            return task_done_callback

                        task.add_done_callback(make_callback(group_id))
                        active_tasks.add(task)
                        break

                # Brief sleep to avoid busy waiting
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error in continuous worker loop: {e}")
                await asyncio.sleep(1)

        if active_tasks:
            print(f"Waiting for {len(active_tasks)} continuous tasks to complete...")
            await asyncio.wait(active_tasks)

        print("Continuous async rollout worker stopped")
    
    def monitor_queue_size(self):
        """监控队列大小并记录到wandb"""
        while self.running:
            try:
                queue_size = self.output_queue.qsize()
                wandb.log({
                    "queue/output_queue_size": queue_size,
                    "queue/queue_utilization": queue_size / self.max_queue_size  
                }, step=self.step_counter)
                self.step_counter += 1
                time.sleep(5)  # 每5秒记录一次
            except Exception as e:
                print(f"Error monitoring queue: {e}")
                time.sleep(5)

    def worker_thread_func(self):
        """Worker function running in independent thread"""
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        """Start continuous work mode"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            print("Started continuous async worker thread")
            self.monitor_thread = threading.Thread(target=self.monitor_queue_size, daemon=True)
            self.monitor_thread.start()
            print("Started queue monitoring thread")

    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        print("Stopped async worker thread")

    def get_completed_groups(self) -> list[tuple]:
        """Get completed sample groups"""
        completed = []
        while True:
            try:
                result = self.output_queue.get_nowait()
                completed.append(result)
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        """Get current output queue size"""
        return self.output_queue.qsize()


async def generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    """
    Simplified asynchronous rollout generation - using global continuous worker
    """
    assert args.rollout_global_dataset

    # Get global worker, which will run continuously
    worker = get_global_worker(args, data_buffer)

    # Simplified: directly use rollout_batch_size as target
    target_data_size = args.rollout_batch_size

    data = []
    completed_groups = {}
    do_print = True

    print(f"Starting async rollout generation for {target_data_size} groups")
    print(f"Global worker queue size: {worker.get_queue_size()}")

    # Main loop: collect results from global worker's output queue
    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0  # Warn if no progress for 30 seconds

    while len(data) < target_data_size:
        # Collect completed results
        completed = worker.get_completed_groups()

        made_progress = False
        for group_id, group in completed:
            completed_groups[group_id] = group
            made_progress = True

        if made_progress:
            last_progress_time = time.time()

        # Process completed groups in order (try to maintain order, but not strict requirement)
        processed_any = False

        # Process all available completed groups
        available_ids = list(completed_groups.keys())
        for group_id in available_ids:
            if len(data) >= target_data_size:
                break

            group = completed_groups.pop(group_id)

            # If any sample in the group was aborted, return the whole group to the data buffer
            # and do not forward it to the training engine.
            try:
                any_aborted = any([sample.status == Sample.Status.ABORTED for sample in group])
            except Exception:
                any_aborted = False

            if any_aborted:
                try:
                    # add back to buffer so it can be retried or handled by buffer policy
                    data_buffer.add_samples([group])
                    print(f"Returned aborted group {group_id} to data buffer", flush=True)
                except Exception as e:
                    print(f"Failed to return aborted group {group_id} to buffer: {e}", flush=True)
                # don't count as processed for training
                continue

            if do_print:
                print(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            # Simplified: directly add samples, no filters used
            data.append(group)
            processed_any = True

        # Check progress
        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            print(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Collected: {len(data)}/{target_data_size}"
            )
            last_progress_time = current_time

        # If no results were processed, brief sleep to avoid busy waiting
        if not processed_any:
            await asyncio.sleep(0.01)

    try:
        print("record tool call counts for analysis")
        tool_time_counts = []
        sample_time_counts = []
        code_call_counts = []
        search_call_counts = []
        metrics_to_log = {}

        for group in data:
            for sample in group:
                if hasattr(sample, 'tool_time'):
                    tool_time_counts.append(sample.tool_time)
                if hasattr(sample, 'sample_time'):
                    sample_time_counts.append(sample.sample_time)
                if hasattr(sample, 'code_call_count'):
                    code_call_counts.append(sample.code_call_count)
                if hasattr(sample, 'search_call_count'):
                    search_call_counts.append(sample.search_call_count)

        if tool_time_counts:
            avg_tool_times = sum(tool_time_counts) / len(tool_time_counts)
            avg_sample_times = sum(sample_time_counts) / len(sample_time_counts) if sample_time_counts else 0.0
            avg_tool_times_ratio_per_sample = [
                t / s if s > 0 else 0.0
                for t, s in zip(tool_time_counts, sample_time_counts)
            ]

            metrics_to_log.update({
                "tool/avg_tool_calls_time": avg_tool_times,
                "tool/avg_sample_time": avg_sample_times,
                "tool/avg_tool_time_ratio_per_sample": sum(avg_tool_times_ratio_per_sample) / len(avg_tool_times_ratio_per_sample) if avg_tool_times_ratio_per_sample else 0.0,
                "tool/avg_code_call_count": sum(code_call_counts) / len(code_call_counts) if code_call_counts else 0.0,
                "tool/avg_search_call_count": sum(search_call_counts) / len(search_call_counts) if search_call_counts else 0.0,
            })
            
        
        tool_call_counts = []
        for group in data:
            for sample in group:
                if hasattr(sample, 'tool_call_count'):
                    tool_call_counts.append(sample.tool_call_count)
        
        if tool_call_counts:
            avg_tool_calls = sum(tool_call_counts) / len(tool_call_counts)
            samples_with_tool_calls = sum(1 for count in tool_call_counts if count > 0)

            metrics_to_log.update({
                "tool/avg_tool_calls_per_sample": avg_tool_calls,
                "tool/total_tool_calls": sum(tool_call_counts),
                "tool/samples_with_tool_calls": samples_with_tool_calls,
            })

        tool_token_counts = []
        response_lengths = []
        for group in data:
            for sample in group:
                if hasattr(sample, 'tool_token_count'):
                    tool_token_counts.append(sample.tool_token_count)
                    response_lengths.append(sample.response_length)

        if tool_token_counts:
            total_tool_tokens = sum(tool_token_counts)
            avg_tool_tokens = total_tool_tokens / len(tool_token_counts)
            per_sample_ratios = [
                t / r if r > 0 else 0.0
                for t, r in zip(tool_token_counts, response_lengths)
            ]
            tool_token_ratio = sum(per_sample_ratios) / len(per_sample_ratios)
            metrics_to_log.update({
                "tool/avg_tool_tokens_per_sample": avg_tool_tokens,
                "tool/tool_token_ratio_in_response": tool_token_ratio,
            })
        
        mismatch_counts = []
        for group in data:
            for sample in group:
                if hasattr(sample, 'mismatch'):
                    mismatch_counts.append(sample.mismatch)
         

        if mismatch_counts:
            total_mismatches = sum(mismatch_counts)
            avg_mismatches = total_mismatches / len(mismatch_counts)
            samples_with_mismatches = sum(1 for m in mismatch_counts if m > 0)

            metrics_to_log.update({
                "debug/total_mismatches": total_mismatches,
                "debug/avg_mismatches_per_sample": avg_mismatches,
                "debug/samples_with_mismatches": samples_with_mismatches,
            })

        # Use a dedicated rollout step axis to avoid conflicts with other threads logging wandb step.
        if metrics_to_log:
            global _wandb_metric_defined
            if not _wandb_metric_defined:
                wandb.define_metric("rollout/step")
                wandb.define_metric("tool/*", step_metric="rollout/step")
                wandb.define_metric("debug/*", step_metric="rollout/step")
                _wandb_metric_defined = True

            metrics_to_log["rollout/step"] = rollout_id
            wandb.log(metrics_to_log)
    except Exception:
        pass
    
    duration = time.time() - start_time
    print(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")

    if data:
        print(
            f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
            flush=True,
        )

    data = sorted(data, key=lambda group: group[0].index)
    return data


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        raise ValueError("Evaluation mode not supported in simple async rollout")

    completed_samples = run(generate_rollout_async(args, rollout_id, data_buffer))
    return completed_samples


# Register exit cleanup function

atexit.register(stop_global_worker)
