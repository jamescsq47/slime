from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


pytest.importorskip("openenv")

MIXED_DIR = Path(__file__).resolve().parents[2] / "examples" / "mixed"
if str(MIXED_DIR) not in sys.path:
    sys.path.insert(0, str(MIXED_DIR))

MODULE_PATH = MIXED_DIR / "tbench2_env" / "server" / "tbench2_env_environment.py"
spec = importlib.util.spec_from_file_location("tbench2_env_server_under_test", MODULE_PATH)
assert spec is not None and spec.loader is not None
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)


def test_truncate_observation_keeps_head_tail_and_bounds_size():
    output = "a" * 5000 + "important error at end"
    truncated = server._truncate_observation(output)

    assert len(truncated) <= server._OBSERVATION_CHAR_LIMIT
    assert truncated.startswith("a" * 100)
    assert truncated.endswith("important error at end")
    assert "[truncated" in truncated


def test_streaming_exec_drains_but_never_retains_unbounded_output():
    class FakeAPI:
        def __init__(self):
            self.created = None

        def exec_create(self, container_id, **kwargs):
            self.created = (container_id, kwargs)
            return {"Id": "exec-1"}

        def exec_start(self, exec_id, stream, demux):
            assert (exec_id, stream, demux) == ("exec-1", True, False)
            return iter((b"a" * 100_000, b"tail-error"))

        def exec_inspect(self, exec_id):
            assert exec_id == "exec-1"
            return {"ExitCode": 0}

    api = FakeAPI()
    container = SimpleNamespace(id="container-1", client=SimpleNamespace(api=api))
    env = server.Tbench2DockerEnvironment()
    env._container = container
    env._workdir = "/app"

    exit_code, output = env._exec_in_container("make test")

    assert exit_code == 0
    assert len(output) <= server._OBSERVATION_CHAR_LIMIT
    assert output.startswith("a" * 100)
    assert output.endswith("tail-error")
    assert api.created[1]["workdir"] == "/app"
    assert "cd /app && make test" in api.created[1]["cmd"][-1]


def test_docker_reset_uses_task_image_workdir(tmp_path):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /app/personal-site\n"
    )
    (task / "instruction.md").write_text("fix the site")
    (task / "task.toml").write_text(
        '[environment]\ndocker_image = "example/task:latest"\n'
    )

    class FakeContainer:
        def put_archive(self, destination, data):
            assert destination == "/task"

        def exec_run(self, cmd, workdir, stdout, stderr):
            assert workdir == "/"
            assert "mkdir -p /task" in cmd[-1]
            return 0, b""

    class FakeImages:
        def get(self, image):
            assert image == "example/task:latest"
            return object()

    class FakeContainers:
        def __init__(self):
            self.kwargs = None

        def run(self, **kwargs):
            self.kwargs = kwargs
            return FakeContainer()

    containers = FakeContainers()
    env = server.Tbench2DockerEnvironment(tasks_dir=str(tmp_path))
    env._docker_client = SimpleNamespace(images=FakeImages(), containers=containers)

    observation = env.reset(task_id="task")

    assert containers.kwargs["working_dir"] == "/app/personal-site"
    assert observation.info["working_directory"] == "/app/personal-site"
