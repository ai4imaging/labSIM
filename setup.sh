#!/usr/bin/env bash
# Repo-root entry point. Implementation lives in scripts/setup.sh so CI and
# the documented path stay one file.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/setup.sh" "$@"
