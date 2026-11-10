"""Reporting contract tests using hand-checkable trajectories, never model inference."""
import csv
import json
import random

import numpy as np
import pytest

from agentir.reporting import RankData, bri_per_target, build_report, paired_bootstrap, statistics


def tiny_data():
    """Create three paired targets with a rescue, a loss and an unchanged case."""
    data=RankData()
    for method,values in {'base':[[20,20,20],[2,2,2],[5,5,5]],'agent':[[20,8,3],[2,12,15],[5,5,5]]}.items():
        for index,ranks in enumerate(values):
            for state,rank in enumerate(ranks):
                data.add(method,f'target{index}',state,rank)
    return data


def test_target_alignment_and_paired_outcomes():
    """Preserve matched identities/order and distinguish rescues from net improvement."""
    arrays,targets,_=tiny_data().align()
    result=statistics(arrays,[('agent','base')],samples=100)
    effect=result['paired_differences']['agent - base']
    assert effect['recall@10'][0]==0
    assert effect['final_rank_better_worse_same']==[1,1,1]
    assert len(result['per_round'])==6 and targets==['target0','target1','target2']
    assert result['final_metrics'][1]['hits@10']==100
    assert result['final_metrics'][1]['recall@10']==pytest.approx(200/3)


def test_missing_targets_and_states_require_explicit_choice():
    """Do not silently pair different cohorts or truncate incomplete trajectories."""
    data=tiny_data()
    data.add('base','extra',0,1)
    with pytest.raises(ValueError,match='allow-intersection'):
        data.align()
    arrays,_,metadata=data.align(allow_intersection=True)
    assert metadata['excluded_targets']==['extra']
    del data.rows['agent','target2',2]
    with pytest.raises(ValueError,match='allow-intersection'):
        data.align()
    assert len(data.align(states=2,allow_intersection=True)[1])==3


def test_bri_matches_hand_calculation_and_single_state():
    """BRI uses natural log of best-so-far one-based ranks, not current ranks or log10."""
    a=np.array([[100,10,20],[1,1,1]])
    expected=(np.log(100)+2*np.log(10)+np.log(10))/4
    assert bri_per_target(a).tolist()==pytest.approx([expected,0])
    assert bri_per_target(a[:,:1]) is None


def test_bootstrap_matches_teammate_exact_algorithm():
    """Use the same target-resampling and percentile index convention as final_results.py."""
    differences=[1,-1,0,1,0,-1,1]
    rng=random.Random(42)
    means=sorted(sum(differences[rng.randrange(7)] for _ in range(7))/7 for _ in range(5000))
    expected=[100*sum(differences)/7,100*means[125],100*means[4875]]
    assert paired_bootstrap(differences,random.Random(42))==expected


def test_supported_loaders_are_equivalent_and_duplicates_guarded(tmp_path):
    """Our long CSV, teammate wide CSV and report JSON must yield the same paired ranks."""
    original=tiny_data()
    rows=[dict(method=m,target=t,round=r,target_rank=v) for (m,t,r),v in original.rows.items()]
    report=tmp_path/'report.json'
    report.write_text(json.dumps({'per_example':rows,'protocol':{'targets':original.targets,'pool':50000}}))
    long=tmp_path/'per_example.csv'
    with long.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    wide=tmp_path/'ranks.csv'
    with wide.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['target','method','round_0','round_1','round_2'])
        for m in ('base','agent'):
            for t in original.targets:
                writer.writerow([t,m]+[original.rows[m,t,r] for r in range(3)])
    for path in (report,long,wide):
        assert RankData().load(str(path)).rows==original.rows
    original.add('agent','target0',0,20)
    with pytest.raises(ValueError,match='Conflicting'):
        original.add('agent','target0',0,99)


def test_export_uses_new_folder_and_single_target_has_no_ci(tmp_path):
    """Debugging pilots may render trajectories but cannot produce a meaningful target bootstrap."""
    data=RankData()
    for m,rs in {'base':[20,15],'agent':[20,3]}.items():
        for r,rank in enumerate(rs):
            data.add(m,'only',r,rank)
    result=build_report(data,tmp_path/'output',samples=100,figures=False)
    assert result['paired_differences']['agent - base']['recall@10']==[100,None,None]
    assert (tmp_path/'output/per_target_ranks.csv').is_file()
    with pytest.raises(FileExistsError):
        build_report(data,tmp_path/'output',samples=100,figures=False)


def test_figures_render_without_gpu_or_private_vm_files(tmp_path):
    """Exercise the full plotting path, including one-state BRI and portable examples."""
    pytest.importorskip('matplotlib')
    output=tmp_path/'figures_run'
    named=RankData()
    for (method,target,state),rank in tiny_data().rows.items():
        named.add('agents_critique' if method=='agent' else 'agents_no_critique',target,state,rank)
    build_report(named,output,samples=100)
    assert (output/'figures/fig3_recall_hits_by_round.png').stat().st_size>1000
    assert (output/'figures/fig5_effect_sizes.pdf').stat().st_size>1000
    assert (output/'figures/fig1_setup.png').stat().st_size>1000
    assert (output/'example_conversations.md').is_file()


def test_saved_reference_export_and_conflicting_pools(tmp_path):
    """Support the teammate's saved GPT reference without silently merging different corpora."""
    path=tmp_path/'saved.json'
    path.write_text(json.dumps({'targets':['one'],'saved_gpt35_ranks':[[5,2,1]]}))
    data=RankData().load(str(path))
    assert data.rows['supplied_plugir','one',2]==1
    data.pool_sizes={100,50000}
    with pytest.raises(ValueError,match='pool sizes'):
        data.align()

