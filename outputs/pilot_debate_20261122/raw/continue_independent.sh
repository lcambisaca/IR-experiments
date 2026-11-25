#!/usr/bin/env bash
set -u
cd ~/pilot_debate_20261122
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw --format=csv -l 1 > gpu_samples_continuations.csv & monitor_pid=$!
for item in agents_debate:0 agents_debate:1 live_plugir_qwen:0 live_plugir_qwen:1; do
 method=${item%:*}; target=${item#*:}
 printf 'START %s ' "$item"; date -u +%FT%TZ
 ~/.venvs/agentir-debate/bin/python -u execute_session.py "$method" "$target" > "${method}_${target}.log" 2>&1
 result=$?
 printf '%s\n' "$result" > "${method}_${target}.exit.txt"
 printf 'FINISHED %s exit=%s ' "$item" "$result"; date -u +%FT%TZ
done
kill "$monitor_pid" 2>/dev/null || true
echo ALL_INDEPENDENT_HISTORIES_ATTEMPTED
