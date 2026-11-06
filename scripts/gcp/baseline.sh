#!/bin/bash
# PlugIR + Qwen baseline (original PlugIR, unchanged) on the first $TOTAL matched targets, 10 rounds.
# Runs in blocks of 30 so partial results are usable; saved conversations are resumed, never regenerated.
# Shuts the VM down when finished or after DEADLINE_HOURS.
cd ~/IR-plugir
TOTAL=${TOTAL:-330}
DEADLINE=$(( $(date +%s) + ${DEADLINE_HOURS:-45} * 3600 ))
for n in $(seq 30 30 $TOTAL); do
  if [ $(date +%s) -ge $DEADLINE ]; then echo "$(date) deadline reached"; break; fi
  echo "$(date) START targets=$n selection=plugir_kl"
  HF_HUB_OFFLINE=1 .venv/bin/python -c "
import run_evaluation as r
r.N_TARGETS, r.N_STATES, r.START_TARGET, r.SELECTION = $n, 11, 0, \"plugir_kl\"
r.main()" > logs/baseline_$n.log 2>&1
  status=$?
  echo "$(date) END targets=$n exit=$status"
  if [ $status -ne 0 ]; then echo "$(date) stopping after failure"; break; fi
done
echo "$(date) baseline finished; shutting down"
sudo shutdown -h +2
