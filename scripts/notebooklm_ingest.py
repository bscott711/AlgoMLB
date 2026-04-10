import os
import sys
import subprocess
from pathlib import Path
import argparse
from loguru import logger

# List of extensions to include
INCLUDE_EXTENSIONS = {
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".sh",
    ".sql",
    ".txt",
    ".json",
    ".ini",
}

# List of directories/filenames to ignore
IGNORE_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    ".antigravity",
}


def is_ignored(path: Path) -> bool:
    for part in path.parts:
        if part in IGNORE_NAMES:
            return True
    return False


def bundle_directory(target_dir: Path) -> str:
    """Recursively collects files and bundles them into a single markdown-formatted string."""
    bundle = []
    bundle.append(f"# Repository Layer: {target_dir.name}\n")
    bundle.append(f"Ingested from: `{target_dir.absolute()}`\n\n")

    files_processed = 0
    for root, dirs, files in os.walk(target_dir):
        # Filter directories in-place for efficiency
        dirs[:] = [d for d in dirs if d not in IGNORE_NAMES]

        for file in files:
            file_path = Path(root) / file
            if is_ignored(file_path):
                continue

            if file_path.suffix.lower() not in INCLUDE_EXTENSIONS:
                continue

            try:
                # Use relative path for the delimiter
                rel_path = file_path.relative_to(target_dir.parent)
                content = file_path.read_text(encoding="utf-8", errors="ignore")

                bundle.append(f"## FILE: {rel_path}\n")
                bundle.append(
                    "```" + (file_path.suffix[1:] if file_path.suffix else "")
                )
                bundle.append(content)
                bundle.append("```\n\n")
                files_processed += 1
            except Exception as e:
                logger.error(f"Could not read {file_path}: {e}")

    logger.info(f"Bundled {files_processed} files from {target_dir}")
    return "\n".join(bundle)


def upload_to_notebooklm(content: str, title: str, notebook_id: str | None = None):
    """Uploads the bundled content to NotebookLM using the CLI."""
    import tempfile

    # Write content to a temporary file to avoid "Argument list too long" errors
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        cmd = ["uv", "run", "notebooklm", "source", "add", tmp_path, "--title", title]
        if notebook_id:
            cmd.extend(["--notebook", notebook_id])

        logger.info(f"Uploading source '{title}' as file {tmp_path}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.success(f"Successfully uploaded '{title}'")
            print(result.stdout)
        else:
            logger.error(f"Failed to upload '{title}': {result.stderr}")
            sys.exit(1)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Bundle and ingest a directory into NotebookLM."
    )
    parser.add_argument(
        "directory", type=str, help="Directory to ingest (e.g., src, tests)"
    )
    parser.add_argument(
        "--title", type=str, required=True, help="Title for the NotebookLM source"
    )
    parser.add_argument("--notebook", type=str, help="Notebook ID (optional)")

    args = parser.parse_args()

    target_path = Path(args.directory)
    if not target_path.exists() or not target_path.is_dir():
        logger.error(f"Directory not found: {args.directory}")
        sys.exit(1)

    bundle_content = bundle_directory(target_path)
    upload_to_notebooklm(bundle_content, args.title, args.notebook)


if __name__ == "__main__":
    main()
