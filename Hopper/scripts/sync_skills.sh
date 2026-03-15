#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$repo_root/.agents/skills"
target_dir="$repo_root/.claude/skills"

if [[ ! -d "$source_dir" ]]; then
  echo "Shared skill source not found: $source_dir" >&2
  exit 1
fi

mkdir -p "$target_dir"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete --exclude '.DS_Store' "$source_dir/" "$target_dir/"
else
  echo "rsync is required to mirror shared skills into .claude/skills" >&2
  exit 1
fi

echo "Synced shared skills from $source_dir to $target_dir"
