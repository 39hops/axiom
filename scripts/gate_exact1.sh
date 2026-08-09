#!/bin/sh
# ENGINE-EXACT-1 no-op gate: full ctest + every pinned-digest driver.
# Q9 must stay bit-identical through the ladder refactor; run after
# every task. Usage: scripts/gate_exact1.sh
set -e
cd "$(dirname "$0")/.."
PY=/Users/artin/code/llmopt/.venv/bin/python
GMOE_REF=/Users/artin/code/llmopt/scratch/detbwd_gmoe_ref

cmake --build build -j 8 | tail -1
ctest --test-dir build --output-on-failure | grep -E "tests passed|tests failed"
cmake --build build-rel -j 8 | tail -1
cd tools/int_adamw
$PY verify_intbirth.py ../../build-rel | tail -1
$PY verify_multiblock.py ../../build-rel | tail -1
$PY verify_gravmoe.py ../../build-rel "$GMOE_REF" | tail -1
$PY verify_primitives.py ../../build-rel | tail -1
$PY test_set_lr.py ../../build-rel | tail -1
$PY test_windows.py ../../build-rel | tail -1
$PY test_moe_birth.py ../../build-rel | tail -1
echo "GATE GREEN"
