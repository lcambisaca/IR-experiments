#!/usr/bin/env bash
set -euo pipefail
mkdir -p ~/pilot_debate_20261122
cd ~/pilot_debate_20261122
printf '%s  %s\n' 8871188e6616dc3610add8261df0a6f397cac371f62da7ac3dcb186e54d1148f "$HOME/debate_pilot_source.tar.gz" | sha256sum -c -
git clone --no-checkout https://github.com/lcambisaca/IR-experiments.git IR-experiments
cd IR-experiments
git checkout 82ed6186d4c5b98d32dff4e949e32d45804b023f
cd ..
tar -xzf ~/debate_pilot_source.tar.gz
cd IR-experiments
python3 - <<'PY'
import json,hashlib,pathlib
m=json.loads(pathlib.Path('TRANSFER_MANIFEST.json').read_text())
bad=[n for n,h in m['files'].items() if hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()!=h]
assert not bad,bad
print('SOURCE_HASHES_VERIFIED',len(m['files']))
PY
git status --short
python3 -m pip install --user uv==0.13.7
~/.local/bin/uv venv --python 3.11 --seed ~/.venvs/agentir-debate
source ~/.venvs/agentir-debate/bin/activate
python -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir -e '.[live,test]'
python -m pip check
python -m pip freeze > ../packages.txt
python -m pytest -q > ../pytest.log 2>&1
nvidia-smi > ../gpu-identity.txt
printf 'ENVIRONMENT_READY\n'
