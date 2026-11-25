"""Validate the final review links, source identity and byte-identical evidence copies."""
import ast
import hashlib
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parent
out=root/'outputs/pilot_20261120'
raw=Path('\\\\?\\'+str(out/'raw'))
def io_path(p):
    s=str(p)
    return p if s.startswith('\\\\?\\') else Path('\\\\?\\'+s)
def sha(p):return hashlib.sha256(io_path(p).read_bytes()).hexdigest()
report=(root/'PILOT_RESULTS_REVIEW.md').read_text(encoding='utf-8')
assert 'Qualitative review to be completed' not in report
assert not re.search(r'\bTODO\b|\bTBD\b',report)
links=[]
for target in re.findall(r'\]\(([^)]+)\)',report):
    if target.startswith(('https://','http://','#')):continue
    p=(root/target).resolve()
    assert p.is_relative_to(root)
    assert io_path(p).is_file(),str(p)
    links.append(target)
pre=json.loads((raw/'pilot-evidence-20261120/preflight.json').read_text())
for filename,expected in pre['source_sha256'].items():
    data=(root/filename).read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(data).hexdigest()==expected,filename
aliases=json.loads((out/'dialogue_aliases.json').read_text())
for item in aliases:
    assert sha(out/item['alias'])==sha(out/item['source'])==item['sha256']
for t in pre['targets']:
    assert sha(raw/'IR-experiments-pilot/data/images'/t['target'])==t['jpeg_sha256']
expected=dict(RUN_LIVE_PLUGIR=True,RUN_MULTI_AGENT=True,USE_PILOT_TARGETS=True,N_STATES=11,
              INCLUDE_SAVED_REFERENCES=False,DEVICE='cuda',LIVE_DEVICE='cuda')
tree=ast.parse((raw/'IR-experiments-pilot/run_evaluation.py').read_text())
settings={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
          and isinstance(n.targets[0],ast.Name) and n.targets[0].id in expected}
assert settings==expected
assert sha(out/'pilot-delivery-20261120.tar.gz')=='0acf872597784e83bca14d37f587facfd8bfdb95fb895aff94231de7b2d0a802'
result=dict(status='passed',local_report_links=len(links),source_files_matching_frozen_preflight=len(pre['source_sha256']),
            checkpoint_aliases_identical=len(aliases),target_jpegs_identical=len(pre['targets']),settings=settings,
            report_sha256=sha(root/'PILOT_RESULTS_REVIEW.md'),archive_sha256=sha(out/'pilot-delivery-20261120.tar.gz'))
(out/'analysis/final_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
