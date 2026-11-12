"""Explicit small caption-only development run; original evaluation runner is unchanged."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from agentir.cache import load_cache, read_json, digest
from agentir.evaluation import matched_queries, evaluate_experiments, save_experiment_results, print_experiment_comparison
from agentir.retrieval import BlipRetriever
from agentir.multi_agent import AgentSettings
from agentir.shared_evidence import prepare_shared_experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-target', type=int, default=1)
    parser.add_argument('--targets', type=int, default=3)
    parser.add_argument('--states', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.targets <= 3 or not 1 <= args.states <= 3 or args.start_target < 0:
        raise ValueError('Development budget: 1..3 targets and 1..3 states.')
    root = Path(__file__).resolve().parent
    config = {k:root/v for k,v in read_json(root/'experiment.json').items()}
    cache = load_cache(config['cache'],config['manifest'])
    pairs = matched_queries(read_json(config['baseline_queries']),read_json(config['supplied_queries']),cache.paths)
    pairs = pairs[args.start_target:args.start_target+args.targets]
    if len(pairs) != args.targets:
        raise ValueError('Insufficient matched targets.')
    torch.manual_seed(1021)
    retriever = BlipRetriever(cache,device='cuda',batch_size=8,threads=4,local_files_only=True)
    bundle = prepare_shared_experiment(root,retriever,pairs,n_states=args.states,
                                     settings=AgentSettings(max_targets=3),offline=True)
    result = evaluate_experiments(retriever,pairs,bundle.methods,rounds=args.states,ks=[1,10,100])
    result['run'] = {'label':'shared_caption_development','settings':vars(args),'metadata':bundle.metadata,
                     'runner_sha256':digest(Path(__file__))}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = save_experiment_results(result,root/'runs/experiments'/stamp)
    for name,content in bundle.artifacts.items():
        (destination/name).write_text(json.dumps(content,indent=2,ensure_ascii=False),encoding='utf-8')
    (destination/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    from agentir.reporting import RankData, build_report
    analysis = RankData()
    analysis.load(str(destination))
    build_report(analysis, destination/'analysis', methods=list(bundle.methods),
                 comparisons=[('shared_roles','shared_neutral_three'),
                              ('shared_roles_critique','shared_roles')], examples=args.targets)
    print_experiment_comparison(result,k=10)
    print('SHARED_RESULTS',destination,flush=True)

if __name__ == '__main__':
    main()
