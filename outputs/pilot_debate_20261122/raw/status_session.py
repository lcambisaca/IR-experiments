import pathlib,json
b=pathlib.Path.home()/'pilot_debate_20261122'; r=b/'IR-experiments/runs'
for pattern in ['three_agents/**/critique_selector_retry_once/*.json','three_agents_debate/**/dialogues/*.json','live_upstream/**/dialogues/*.json']:
 for p in r.glob(pattern):
  d=json.loads(p.read_text());s=d.get('states',[]);print(p.relative_to(r),d.get('status'),len(s),'states','rank',s[-1].get('target_rank') if s else None,d.get('last_error',''));print('LAST_QUESTION:',s[-1].get('question') if s else None)
print('PRIMARY_EXIT:',(b/'primary_exit_code.txt').read_text().strip() if (b/'primary_exit_code.txt').exists() else 'running')
print('QWEN_CACHE_RECORDS:',sum(1 for p in r.glob('**/qwen_calls/*.json')))
