#!/usr/bin/env bash
set -u
cd ~/pilot_debate_20261122
while kill -0 1716 2>/dev/null; do sleep 5; done
if ! test -f preflight.json; then echo 'PREFLIGHT_FAILED: generation not started'; exit 2; fi
if test -d IR-experiments/runs; then echo 'Existing run folder found; refusing unintended checkpoint reuse'; exit 2; fi
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw --format=csv -l 1 > gpu_samples.csv & monitor_pid=$!
printf 'START '; date -u +%FT%TZ
~/.venvs/agentir-debate/bin/python -u execute_session.py > primary_runner.log 2>&1
result=$?
kill "$monitor_pid" 2>/dev/null || true
printf '%s\n' "$result" > primary_exit_code.txt
printf 'PRIMARY_FINISHED exit=%s ' "$result"; date -u +%FT%TZ
