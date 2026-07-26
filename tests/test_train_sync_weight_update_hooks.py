from __future__ import annotations

from types import SimpleNamespace

import train as train_module


class RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def test_sync_train_notifies_rollout_manager_around_weight_updates(monkeypatch):
    events = []

    class RolloutManager:
        get_metrics_router_addr = RemoteMethod(lambda: None)
        generate = RemoteMethod(lambda rollout_id: events.append(("generate", rollout_id)) or "rollout-data")
        before_weight_update = RemoteMethod(lambda version: events.append(("before_weight_update", version)))
        after_weight_update = RemoteMethod(lambda version: events.append(("after_weight_update", version)))
        dispose = RemoteMethod(lambda: events.append(("dispose", None)))

    class ActorModel:
        def update_weights(self):
            events.append(("update_weights", None))

        def async_train(self, rollout_id, data):
            events.append(("train", rollout_id))
            assert data == "rollout-data"
            return None

        def clear_memory(self):
            events.append(("clear_memory", None))

    manager = RolloutManager()
    actor = ActorModel()
    monkeypatch.setattr(train_module.ray, "get", lambda value: value)
    monkeypatch.setattr(train_module, "create_placement_groups", lambda args: {"rollout": object()})
    monkeypatch.setattr(train_module, "create_rollout_manager", lambda args, pg: (manager, 1))
    monkeypatch.setattr(train_module, "create_training_models", lambda args, pgs, rollout: (actor, None))
    monkeypatch.setattr(train_module, "configure_logger", lambda: None)
    monkeypatch.setattr(train_module, "init_tracking", lambda args: None)
    monkeypatch.setattr(train_module, "update_tracking_open_metrics", lambda args, addr: None)
    monkeypatch.setattr(train_module, "finish_tracking", lambda args: None)
    monkeypatch.setattr(train_module, "should_run_periodic_action", lambda *args, **kwargs: False)

    args = SimpleNamespace(
        offload_rollout=False,
        offload_train=False,
        critic_train_only=False,
        check_weight_update_equal=False,
        num_rollout=4,
        start_rollout_id=3,
        eval_interval=None,
        save_interval=None,
        use_critic=False,
        rollout_global_dataset=False,
    )

    train_module.train(args)

    assert events == [
        ("update_weights", None),
        ("after_weight_update", 3),
        ("generate", 3),
        ("train", 3),
        ("clear_memory", None),
        ("before_weight_update", 3),
        ("update_weights", None),
        ("after_weight_update", 4),
        ("dispose", None),
    ]
