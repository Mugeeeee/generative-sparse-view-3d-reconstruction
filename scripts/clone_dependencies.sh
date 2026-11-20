#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVISION_FILE="$PROJECT_ROOT/third_party/revisions.tsv"
REPO_DIR="$PROJECT_ROOT/third_party/repos"

mkdir -p "$REPO_DIR"

while IFS=$'\t' read -r name url commit; do
    [ -n "$name" ] || continue

    target="$REPO_DIR/$name"

    echo "===== $name ====="

    if [ -d "$target/.git" ]; then
        echo "Repository already exists: $target"
        git -C "$target" fetch --all --tags
    else
        git clone "$url" "$target"
    fi

    git -C "$target" checkout "$commit"
    git -C "$target" submodule update --init --recursive
done < "$REVISION_FILE"

echo
echo "All third-party repositories are ready in:"
echo "$REPO_DIR"
