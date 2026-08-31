#!/usr/bin/env python3
"""Cross-platform GGUF model weights downloader for kimodo.cpp using huggingface_hub."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

ORG = "LocalAI-io"
TEXT_REPO_DEFAULT = f"{ORG}/Llama-3-Kimodo-GGML"

MODEL_MAP = {
    "soma-rp-v1.1": (f"{ORG}/Kimodo-SOMA-RP-v1.1-GGML", "models/kimodo-soma-rp-v1.1-f32.gguf"),
    "soma-seed-v1.1": (f"{ORG}/Kimodo-SOMA-SEED-v1.1-GGML", "models/kimodo-soma-seed-v1.1-f32.gguf"),
    "g1-rp-v1": (f"{ORG}/Kimodo-G1-RP-v1-GGML", "models/kimodo-g1-rp-v1-f32.gguf"),
    "g1-seed-v1": (f"{ORG}/Kimodo-G1-SEED-v1-GGML", "models/kimodo-g1-seed-v1-f32.gguf"),
}

def verify_file(file_path: Path, expected_bytes: int, expected_sha256: str):
    if not file_path.is_file():
        raise SystemExit(f"File missing: {file_path}")
    if file_path.stat().st_size != expected_bytes:
        raise SystemExit(f"Size mismatch for {file_path}: expected {expected_bytes}, got {file_path.stat().st_size}")
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != expected_sha256:
        raise SystemExit(f"Checksum mismatch for {file_path}")

def download_and_verify(repo: str, revision: str, output_dir: Path, patterns: list[str]):
    manifest_local = hf_hub_download(
        repo_id=repo,
        filename="MANIFEST.json",
        revision=revision,
        local_dir=output_dir / ".kimodo-manifests" / repo.replace("/", "__")
    )
    manifest = json.loads(Path(manifest_local).read_text(encoding="utf-8"))
    if manifest.get("format") != "kimodo-gguf-manifest-v1":
        raise SystemExit(f"Unsupported or malformed manifest in {repo}")

    for entry in manifest.get("files", []):
        rel_path = entry.get("path", "")
        p = Path(rel_path)
        if p.is_absolute() or ".." in p.parts or p.suffix != ".gguf":
            continue
        
        matches = False
        for pat in patterns:
            if pat.endswith("*"):
                prefix = pat[:-1]
                if rel_path.startswith(prefix):
                    matches = True
                    break
            elif rel_path == pat:
                matches = True
                break
        
        if not matches:
            continue

        print(f"Downloading {rel_path} from {repo}...")
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=rel_path,
            revision=revision,
            local_dir=output_dir
        )
        print(f"Verifying {rel_path}...")
        verify_file(Path(downloaded), entry.get("bytes"), entry.get("sha256"))

def main():
    parser = argparse.ArgumentParser(description="Download kimodo.cpp GGUF weights")
    parser.add_argument("--output", default=".", help="Output directory root")
    parser.add_argument("--model", action="append", choices=list(MODEL_MAP.keys()), help="Model(s) to download")
    parser.add_argument("--motion-repo", help="Override motion repo")
    parser.add_argument("--text-repo", default=TEXT_REPO_DEFAULT, help="Text encoder repo")
    parser.add_argument("--revision", default="main", help="Git revision")
    parser.add_argument("--motion-only", action="store_true", help="Download only motion weights, skip text bundle")
    args = parser.parse_args()

    models = args.model if args.model else ["soma-rp-v1.1"]
    output_path = Path(args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    for model in models:
        default_repo, default_file = MODEL_MAP[model]
        repo = args.motion_repo if args.motion_repo else default_repo
        print(f"--- Downloading Motion Model: {model} ({repo}) ---")
        download_and_verify(repo, args.revision, output_path, [default_file])

    if not args.motion_only:
        print(f"--- Downloading Text Encoder Bundle ({args.text_repo}) ---")
        download_and_verify(args.text_repo, args.revision, output_path, ["generated/llm2vec-text-bundle/*"])

    print("\nAll requested Kimodo GGUF models downloaded and verified successfully!")

if __name__ == "__main__":
    main()
