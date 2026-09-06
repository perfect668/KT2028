#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/surgen_hemicore
PYTHONPATH=src /root/miniconda3/envs/liverct/bin/python -m py_compile src/MSRLSelect/formal_dense.py
PYTHONPATH=src /root/miniconda3/envs/liverct/bin/python -m MSRLSelect.rl_core.test_core
# Independent hand/action audit was executed as an inline Python command; no training/evaluation entrypoint was run.
