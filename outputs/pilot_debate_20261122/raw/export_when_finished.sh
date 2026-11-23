#!/usr/bin/env bash
set -euo pipefail
while kill -0 1997 2>/dev/null; do sleep 5; done
cd ~/pilot_debate_20261122
cp ~/setup_debate_pilot.sh setup_debate_pilot.sh
cp ~/pilot_debate_setup.log setup.log
python3 - <<'PY'
import pathlib,json,hashlib,subprocess
r=pathlib.Path('IR-experiments');m=json.loads((r/'TRANSFER_MANIFEST.json').read_text())
bad=[n for n,h in m['files'].items() if hashlib.sha256((r/n).read_bytes()).hexdigest()!=h]
assert not bad,bad
info={'all_original_files_unchanged':True,'verified_files':len(m['files']),'commit':subprocess.check_output(['git','-C',str(r),'rev-parse','HEAD']).decode().strip(),'git_status':subprocess.check_output(['git','-C',str(r),'status','--short']).decode()}
pathlib.Path('final_source_verification.json').write_text(json.dumps(info,indent=2))
files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in pathlib.Path('.').rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and 'data/cache' not in str(p)}
pathlib.Path('ARTIFACT_MANIFEST.json').write_text(json.dumps(files,indent=2))
print('FINAL_SOURCE_VERIFIED',len(m['files']),'ARTIFACT_FILES',len(files),flush=True)
PY
cd ~
tar --exclude='pilot_debate_20261122/IR-experiments/.git' --exclude='pilot_debate_20261122/IR-experiments/data/cache' --exclude='__pycache__' --exclude='.pytest_cache' -czf pilot_debate_delivery.tar.gz pilot_debate_20261122
sha256sum pilot_debate_delivery.tar.gz
printf 'DELIVERY_ARCHIVE_READY\n'
