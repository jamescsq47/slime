"""Pinned DeepSeek-V4-Flash download and read-only compatibility inspection.

This is NOT a DeepSeek V4 model implementation or a GPU inference launcher.
It does not execute downloaded model Python files or install dependencies.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO_ID = "deepseek-ai/DeepSeek-V4-Flash"
REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
SLIME_ROOT = Path(__file__).resolve().parents[2]
SGLANG_ROOT = SLIME_ROOT.parent / "sglang"
DEFAULT_ROOT = SLIME_ROOT / "downloads" / "deepseek-v4-flash" / REVISION


def command_output(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"rc": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"rc": -1, "error": str(exc)}


def defined_model_classes(root):
    """Source inspection, not an import that initializes GPU-specific modules."""
    classes = set()
    for path in (root / "python/sglang/srt/models").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        classes.update(node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    return classes


def assess(config, classes):
    architectures = config.get("architectures", [])
    native = bool(architectures) and all(name in classes for name in architectures)
    deepseek_v4 = config.get("model_type") == "deepseek_v4"
    blockers = []
    if not native:
        blockers.append("Current SGLang checkout has no native implementation for " + repr(architectures)
                        + "; automatic Transformers fallback is not proof of TP/backend compatibility")
    if deepseek_v4:
        blockers.append("Current multi-node launcher accepts dense Qwen3 only; do not bypass its model guard")
        blockers.append("V4 compressed-attention snapshot requires a dedicated adapter; MHA K/V spans are not compatible")
    return {"architectures": architectures, "model_type": config.get("model_type"),
            "native_model_class_present": native,
            "expert_dtype": config.get("expert_dtype"),
            "quantization_config": config.get("quantization_config"),
            "compression_ratios": sorted(set(config.get("compress_ratios", []))),
            "single_node_tp8_model_run": "blocked" if not native else "not_validated",
            "custom_bidirectional_pd": "blocked" if deepseek_v4 else "not_validated",
            "blockers": blockers,
            "note": "Eight mock shards or a NCCL all-reduce do not validate this model or its KV state"}


def download(root, weights):
    # Keep downloaded payloads and caches under the authorized clone, not $HOME.
    os.environ.setdefault("HF_HOME", str(root.parent / "hub-cache"))
    os.environ.setdefault("HF_XET_CACHE", str(root.parent / "xet-cache"))
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    info = api.model_info(REPO_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError("Model revision changed unexpectedly")
    files = [{"name": item.rfilename, "size": item.size} for item in info.siblings]
    weight_files = [item for item in files if item["name"].endswith(".safetensors")]
    if any(item["size"] is None for item in weight_files):
        raise RuntimeError("Missing weight size metadata")
    remaining = sum(max(0, item["size"] - (
        (root / item["name"]).stat().st_size if (root / item["name"]).is_file() else 0
    )) for item in weight_files)
    if weights and shutil.disk_usage(root).free < remaining + 10 * 1024**3:
        raise RuntimeError("Not enough disk for pinned weights plus 10 GiB safety margin")
    manifest = {"repo_id": REPO_ID, "revision": REVISION, "files": files,
                "weight_bytes": sum(item["size"] for item in weight_files)}
    (root / "download-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"action": "weights" if weights else "metadata", "directory": str(root),
                      "revision": REVISION, "weight_bytes": manifest["weight_bytes"]}), flush=True)
    patterns = ["*.json", "README.md", "LICENSE", "encoding/**", "inference/**"]
    if weights:
        patterns.append("*.safetensors")
    snapshot_download(REPO_ID, revision=REVISION, local_dir=root,
                      allow_patterns=patterns, max_workers=4)
    if weights:
        missing = [item["name"] for item in weight_files
                   if not (root / item["name"]).is_file()
                   or (root / item["name"]).stat().st_size != item["size"]]
        if missing:
            raise RuntimeError("Incomplete weights: " + repr(missing))
        (root / "weights-complete.json").write_text(json.dumps({
            "revision": REVISION, "files": len(weight_files), "size_checked": True,
            "model_loaded": False}, indent=2))


def inspect(root):
    config = json.loads((root / "config.json").read_text())
    result = assess(config, defined_model_classes(SGLANG_ROOT))
    result.update(repo_id=REPO_ID, revision=REVISION, model_directory=str(root),
                  weights_complete=(root / "weights-complete.json").exists(),
                  sglang_revision=command_output(["git", "-C", str(SGLANG_ROOT), "rev-parse", "HEAD"]),
                  gpus=command_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,compute_cap",
                                       "--format=csv"]),
                  gpu_processes=command_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                                "--format=csv"]))
    target = root / "compatibility.json"
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("Report:", target)
    return 2 if result["blockers"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("metadata", "download", "inspect"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.model_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.action in {"download", "metadata"}:
        download(root, args.action == "download")
    else:
        raise SystemExit(inspect(root))


if __name__ == "__main__":
    main()
