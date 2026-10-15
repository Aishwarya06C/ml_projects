#!/usr/bin/env bash
# Usage: bash push_to_github.sh https://github.com/<you>/<repo>.git
set -e
cd "$(dirname "$0")"
[ -z "$1" ] && { echo "Usage: bash push_to_github.sh <github-repo-url>"; exit 1; }
git init -q
git add -A
git commit -q -m "Add 15 machine learning projects" || true
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$1"
git push -u origin main
