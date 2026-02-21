#!/usr/bin/env bash
set -euo pipefail

# This script is a template. COLMAP builds vary depending on CUDA toolkit availability in WSL.
# Recommended: pin COLMAP commit and record it in environment.md.

COLMAP_DIR="${1:-$HOME/colmap}"
echo "COLMAP_DIR=$COLMAP_DIR"

echo "If you already have COLMAP built with CUDA, just record:"
echo "  colmap -h | head -n 3"
echo "and the git commit:"
echo "  git -C $COLMAP_DIR rev-parse HEAD"