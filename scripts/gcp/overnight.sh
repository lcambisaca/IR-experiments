#!/bin/bash
# Alternate original PlugIR and the agent in blocks of 10 targets (first 300 matched targets, 10 rounds).
# Each block resumes saved conversations, so earlier targets are not regenerated.
# Stops starting new blocks after DEADLINE_HOURS, then shuts the VM down (stopped VMs cost only disk).
cd ~/IR-plugir
DEADLINE=$(( $(date +%s) + ${DEADLINE_HOURS:-7} * 3600 ))
for n in $(seq 10 10 300); do
  for sel in plugir_kl probe_split; do
    if [ $(date +%s) -ge $DEADLINE ]; then echo "$(date) deadline reached"; break 3; fi
    echo "$(date) START targets=$n selection=$sel"
    HF_HUB_OFFLINE=1 .venv/bin/python -c "
import run_evaluation as r
r.N_TARGETS, r.N_STATES, r.START_TARGET, r.SELECTION = $n, 11, 0, \"$sel\"
r.main()" > logs/${sel}_$n.log 2>&1
    status=$?
    echo "$(date) END targets=$n selection=$sel exit=$status"
    if [ $status -ne 0 ]; then echo "$(date) stopping after failure"; break 3; fi
  done
done
echo "$(date) overnight run finished; shutting down"
sudo shutdown -h +2
