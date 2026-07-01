#!/usr/bin/env bash
# earshot test suite: python unit tests + bash integration tests (stubbed).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rc=0

echo "=== python unit tests ==="
python3 -m unittest discover -s "${HERE}" || rc=1

echo
echo "=== bash integration tests ==="
bash "${HERE}/test_bash.sh" || rc=1

exit "${rc}"
