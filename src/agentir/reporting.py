"""Portable analysis of saved rank trajectories; never loads a model or runs retrieval.

Reporting conventions follow teammate branch 51be8325b2aaa53281c91a8250b0fac547ccffff:
final Recall@1/10/100, cumulative Hits@10, lower median rank, paired target bootstrap,
mean Recall@10 over post-caption rounds, better/worse/same, and per-round figures.
BRI is additionally computed as the original natural-log best-rank trapezoidal mean.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random

import numpy as np

LABELS = {"caption_only": "Original caption", "raw_dialogue": "Saved raw dialogue",
    "supplied_plugir": "Saved PlugIR (GPT-3.5)", "live_plugir_qwen": "Live PlugIR + Qwen",
    "agents_no_critique": "Three agents, no critique", "agents_critique": "Three agents + critique",
    "plugir": "Live PlugIR + Qwen", "agent_v2": "Teammate agent v2",
    "plugir+mix": "PlugIR + caption mix", "agent_v2+mix": "Agent v2 + caption mix"}


class RankData:
    """Keep validated ranks, optional conversation text, target order and source provenance."""

    def __init__(self):
        """Initialize a sparse method/target/state table and insertion-order target list."""
        self.rows, self.details, self.targets, self.sources, self.pool_sizes = {}, {}, [], [], set()

    def add(self, method, target, state, rank, detail=None):
        """Reject inconsistent duplicate records; never count one target twice."""
        if not isinstance(method, str) or not method or not isinstance(target, str) or not target:
            raise ValueError("Each rank requires a nonempty method and target ID.")
        if isinstance(state, bool) or isinstance(rank, bool) or int(state) != float(state) or int(rank) != float(rank):
            raise ValueError("States/ranks must be integers.")
        state, rank = int(state), int(rank)
        if state < 0 or rank < 1:
            raise ValueError("States start at zero and ranks start at one.")
        key = method, target, state
        if key in self.rows and self.rows[key] != rank:
            raise ValueError(f"Conflicting saved ranks for {key}.")
        self.rows[key] = rank
        if detail:
            self.details.setdefault(key, {}).update(detail)
        if target not in self.targets:
            self.targets.append(target)

    def load(self, source):
        """Read our report/long CSV, teammate wide CSV, or explicitly named compact conversation JSON."""
        method_override = None
        if "=" in source:
            method_override, source = source.split("=", 1)
        path = Path(source)
        if path.is_dir():
            path = path / "report.json"
        raw = path.read_bytes()
        self.sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                             "method_override": method_override})
        if path.suffix.lower() == ".csv":
            records = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
            if not records:
                raise ValueError("Empty rank CSV.")
            if "target_rank" in records[0]:
                for row in records:
                    self.add(method_override or row["method"], row["target"], row["round"], row["target_rank"],
                             {"query": row.get("query", "")})
            else:
                columns = sorted((c for c in records[0] if c.startswith("round_")), key=lambda c:int(c[6:]))
                if not columns:
                    raise ValueError("Use per_example.csv or per_target_ranks.csv, not aggregate metrics.")
                for row in records:
                    for column in columns:
                        self.add(method_override or row["method"], row["target"], int(column[6:]), row[column])
        else:
            data = json.loads(raw.decode("utf-8-sig"))
            if "per_example" in data:
                protocol = data.get("protocol", {})
                if protocol.get("pool"):
                    self.pool_sizes.add(int(protocol["pool"]))
                for target in protocol.get("targets", []):
                    if target not in self.targets:
                        self.targets.append(target)
                for row in data["per_example"]:
                    self.add(method_override or row["method"], row["target"], row["round"], row["target_rank"],
                             {"query": row.get("query", "")})
            elif "conversations" in data:
                if not method_override:
                    raise ValueError("Compact conversations need an explicit METHOD=path.json input.")
                self._conversations(method_override, data["conversations"])
            elif "saved_gpt35_ranks" in data:
                targets, trajectories = data["targets"], data["saved_gpt35_ranks"]
                if len(targets) != len(trajectories):
                    raise ValueError("Saved reference targets and rank trajectories differ in length.")
                for target, trajectory in zip(targets, trajectories):
                    for state, rank in enumerate(trajectory):
                        self.add(method_override or "supplied_plugir",target,state,rank)
            elif isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
                for method, cases in data.items():
                    self._conversations(method_override or method, cases)
            else:
                raise ValueError("Unsupported rank input schema.")
        return self

    def _conversations(self, method, cases):
        """Extract public compact exports or our prepared agent dialogue artifacts."""
        for case in cases:
            target = case.get("target", case.get("identity", {}).get("target"))
            for state in case["states"]:
                self.add(method, target, state["round"], state["target_rank"],
                         {k:state[k] for k in ("query", "question", "answer") if k in state})

    def align(self, methods=None, *, states=None, target_order=None, allow_intersection=False):
        """Align complete trajectories on target IDs; require explicit permission for exclusions."""
        methods = list(dict.fromkeys(m for m, _, _ in self.rows)) if methods is None else list(methods)
        if not methods or len(methods) != len(set(methods)):
            raise ValueError("Select distinct existing methods.")
        available = {m for m, _, _ in self.rows}
        if set(methods) - available:
            raise ValueError(f"Unknown methods: {set(methods)-available}")
        if len(self.pool_sizes) > 1:
            raise ValueError("Reported candidate pool sizes differ; do not combine these runs.")
        per_method = {m:{t for method,t,_ in self.rows if method==m} for m in methods}
        union = set.union(*per_method.values())
        maximums = {max(r for method,t,r in self.rows if method==m) for m in methods}
        if states is None:
            if len(maximums) != 1:
                raise ValueError("State counts differ; specify --states explicitly to compare a common prefix.")
            states = maximums.pop()+1
        if type(states) is not int or not 1 <= states <= 11:
            raise ValueError("Use 1..11 states, including the initial caption.")
        complete = {t for t in union if all((m,t,r) in self.rows for m in methods for r in range(states))}
        if not complete:
            raise ValueError("No complete matched targets across the selected methods.")
        excluded = sorted(union-complete)
        if excluded and not allow_intersection:
            raise ValueError("Methods have different/incomplete targets. Inspect them, then explicitly use --allow-intersection.")
        order = self.targets if target_order is None else target_order
        if len(order) != len(set(order)) or complete-set(order):
            raise ValueError("Target order must contain every matched target exactly once (extra IDs are allowed).")
        targets = [t for t in order if t in complete]
        arrays = {m:np.array([[self.rows[m,t,r] for r in range(states)] for t in targets], dtype=np.int64) for m in methods}
        pool = next(iter(self.pool_sizes), 50000)
        if any((a>pool).any() for a in arrays.values()):
            raise ValueError("A rank exceeds the stated candidate pool.")
        return arrays, targets, {"excluded_targets":excluded,"states":states,"pool":pool,
                                 "pool_source":"input report" if self.pool_sizes else "assumed 50000; verify input provenance",
                                 "order": "explicit supplied target order" if target_order is not None else "input order; not necessarily execution order"}


def bri_per_target(ranks):
    """Original PlugIR BRI on one-based ranks; one state has no interaction integral."""
    if ranks.shape[1] == 1:
        return None
    logs = np.log(np.minimum.accumulate(ranks,axis=1).astype(float))
    return ((logs[:,:-1]+logs[:,1:])/2).mean(axis=1)


def paired_bootstrap(difference, rng, *, samples=5000, scale=100.):
    """Resample target-level paired differences using teammate's percentile-index convention."""
    difference = [float(x) for x in difference]
    n = len(difference)
    point = scale*sum(difference)/n
    if n < 2:
        return [point,None,None]
    means = sorted(sum(difference[rng.randrange(n)] for _ in range(n))/n for _ in range(samples))
    return [point,scale*means[int(.025*samples)],scale*means[int(.975*samples)]]


def statistics(arrays, comparisons, *, samples=5000, seed=42):
    """Compute matched per-round tables and paired effect sizes; rounds are not independent samples."""
    if type(samples) is not int or samples < 100:
        raise ValueError("Use at least 100 bootstrap resamples (default 5000).")
    rows, final, effects = [], [], {}
    for method,a in arrays.items():
        best = np.minimum.accumulate(a,axis=1)
        bri = bri_per_target(a)
        for r in range(a.shape[1]):
            row = {"method":method,"round":r,"n_targets":len(a),
                **{f"recall@{k}":100*float((a[:,r]<=k).mean()) for k in (1,10,100)},
                "hits@10":100*float((best[:,r]<=10).mean()),
                "median_rank":int(np.sort(a[:,r])[(len(a)-1)//2]),
                "mean_rank":float(a[:,r].mean())}
            rows.append(row)
        final.append({**rows[-1],"BRI":None if bri is None else float(bri.mean())})
    rng = random.Random(seed)
    for left,right in comparisons:
        if left not in arrays or right not in arrays or left==right:
            raise ValueError("Comparison methods must be distinct selected methods.")
        a,b = arrays[left],arrays[right]
        label = left+" - "+right
        if label in effects:
            raise ValueError("Duplicate comparison.")
        # Preserve teammate's metric/RNG order so the same paired inputs reproduce its ranges.
        diffs = {"recall@1":(a[:,-1]<=1).astype(float)-(b[:,-1]<=1),
                 "recall@10":(a[:,-1]<=10).astype(float)-(b[:,-1]<=10),
                 "hits@10":(a<=10).any(axis=1).astype(float)-(b<=10).any(axis=1)}
        if a.shape[1]>1:
            diffs["recall@10_avg_post_caption"] = (a[:,1:]<=10).mean(1)-(b[:,1:]<=10).mean(1)
        record = {k:paired_bootstrap(v,rng,samples=samples) for k,v in diffs.items()}
        record["final_rank_better_worse_same"] = [int((a[:,-1]<b[:,-1]).sum()),int((a[:,-1]>b[:,-1]).sum()),int((a[:,-1]==b[:,-1]).sum())]
        ba,bb = bri_per_target(a),bri_per_target(b)
        if ba is not None:
            record["BRI_difference"] = paired_bootstrap(ba-bb,random.Random(f"{seed}:bri:{label}"),samples=samples,scale=1.)
        effects[label] = record
    return {"per_round":rows,"final_metrics":final,"paired_differences":effects}


def write_csv(path, rows):
    """Save deterministic spreadsheet-friendly tables without a dataframe dependency."""
    if not rows:
        return
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(arrays, targets, result, comparisons, destination, *, seed=42, examples=3, details=None):
    """Create teammate-style curves, accumulating effects, forest plot, final bars and rank examples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'figure.facecolor':'#fcfcfb',
        'axes.facecolor':'#fcfcfb','savefig.facecolor':'#fcfcfb','lines.linewidth':2})
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    colours = ['#2a78d6','#eb6834','#1baf7a','#eda100','#8a8984','#8b5fbf','#297a7b']
    methods = list(arrays)
    rounds = np.arange(next(iter(arrays.values())).shape[1])
    n = len(targets)

    def save(fig,name):
        """Export PNG for review and vector PDF for the report, then free the figure."""
        fig.savefig(destination/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(destination/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for method,colour in zip(methods,itertools.cycle(colours)):
        table = [r for r in result['per_round'] if r['method']==method]
        for ax,key,title in zip(axes,('recall@10','hits@10'),('Recall@10 by round','Hits@10 by round')):
            ys = [r[key] for r in table]
            ax.plot(rounds,ys,'o-',label=LABELS.get(method,method),color=colour,markersize=3)
            ax.set(title=title,xlabel='Round (0 = original caption)',ylabel='Targets (%)',ylim=(0,103))
            ax.set_xticks(rounds)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=min(3,len(methods)),frameon=False)
    fig.suptitle(f'Same {n} matched targets for every method')
    save(fig,'fig3_recall_hits_by_round')

    if comparisons:
        left,right = comparisons[0]
        d=(arrays[left][:,-1]<=10).astype(float)-(arrays[right][:,-1]<=10)
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        start=min(10,n)
        xs=np.unique(np.linspace(start,n,min(40,n-start+1),dtype=int))
        mean,lo,hi=[],[],[]
        rng=np.random.default_rng(seed)
        for count in xs:
            subset=d[:count]
            mean.append(100*float(subset.mean()))
            if count>1:
                boot=subset[rng.integers(0,count,size=(2000,count))].mean(1)*100
                lo.append(float(np.quantile(boot,.025)));hi.append(float(np.quantile(boot,.975)))
            else:
                lo.append(np.nan);hi.append(np.nan)
        ax.fill_between(xs,lo,hi,color=colours[1],alpha=.17,label='Pointwise 95% bootstrap interval')
        ax.plot(xs,mean,color=colours[1],label=LABELS.get(left,left)+' − '+LABELS.get(right,right))
        ax.axhline(0,color='#52514e',lw=1)
        ax.set(title='Effect as matched targets accumulate (descriptive)',xlabel='Number of targets in recorded input order',
               ylabel=f'Recall@10 difference at round {rounds[-1]} (percentage points)')
        ax.legend(fontsize=8,frameon=False)
        save(fig,'fig4_effect_over_targets')

        fig,ax=plt.subplots(figsize=(11,max(3,.65*len(comparisons)+1.5)),layout='constrained')
        names=[]
        for i,(left,right) in enumerate(comparisons):
            label=left+' - '+right
            point,low,high=result['paired_differences'][label]['recall@10']
            names.append(LABELS.get(left,left)+' − '+LABELS.get(right,right))
            if low is not None:
                ax.plot([low,high],[i,i],color=colours[1],lw=3)
                ax.annotate(f'{point:+.1f} [{low:+.1f}, {high:+.1f}]',(high,i),xytext=(6,0),textcoords='offset points',va='center',fontsize=8)
            else:
                ax.annotate(f'{point:+.1f}; CI unavailable (n=1)',(point,i),xytext=(6,0),textcoords='offset points',va='center',fontsize=8)
            ax.plot(point,i,'o',color=colours[1])
        ax.axvline(0,color='#52514e',lw=1)
        ax.set_yticks(range(len(names)),names)
        ax.margins(x=.35)
        ax.set(title=f'Paired final-round effects ({n} targets)',xlabel='Recall@10 difference (percentage points); 95% paired bootstrap interval')
        ax.grid(axis='y',visible=False)
        save(fig,'fig5_effect_sizes')

    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    final=result['final_metrics']
    for ax,key,title in zip(axes.flat,('recall@1','recall@10','recall@100','hits@10','median_rank','BRI'),
                           ('Recall@1 (%)','Recall@10 (%)','Recall@100 (%)','Hits@10 (%)','Median rank (lower is better)','BRI (lower is better)')):
        vals=[r[key] for r in final]
        if all(v is None for v in vals):
            ax.text(.5,.5,'Undefined for caption-only evaluation',ha='center',va='center',transform=ax.transAxes)
        else:
            bars=ax.bar(range(len(methods)),[0 if v is None else v for v in vals],color=[colours[i%len(colours)] for i in range(len(methods))])
            ax.bar_label(bars,fmt='%.2f' if key=='BRI' else '%.1f',fontsize=8,padding=3)
            ax.margins(y=.2)
            if key in ('recall@1','recall@10','recall@100','hits@10'):
                ax.set_ylim(0,110)
        ax.set_xticks(range(len(methods)),[LABELS.get(m,m) for m in methods],rotation=30,ha='right',fontsize=7)
        ax.set_title(title)
    fig.suptitle(f'Final metrics, round {rounds[-1]}, n={n}')
    save(fig,'fig_final_metrics')

    chosen=[]
    if comparisons:
        left,right=comparisons[0]
        delta=arrays[right][:,-1]-arrays[left][:,-1]
        candidates=[int(delta.argmax()),int(delta.argmin()),int(np.argsort(np.abs(delta))[0])]
    else:
        candidates=list(range(n))
    for index in candidates+list(range(n)):
        if index not in chosen:
            chosen.append(index)
        if len(chosen)>=min(examples,n):
            break
    example_text=['# Illustrative target trajectories','',
        'Examples include large final-rank gains/losses and a small-change case; they are illustrative, not a representative sample.','']
    for count,index in enumerate(chosen,1):
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        for method,colour in zip(methods,itertools.cycle(colours)):
            ax.plot(rounds,arrays[method][index],'o-',label=LABELS.get(method,method),color=colour,markersize=3)
        ax.set_yscale('log');ax.invert_yaxis();ax.axhline(10,color='#8a8984',ls='--',lw=1)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set(title=targets[index],xlabel='Round',ylabel='Target rank (log scale; 1 = best)')
        ax.set_xticks(rounds);ax.legend(fontsize=8,frameon=False,loc='upper left',bbox_to_anchor=(1.01,1))
        save(fig,f'example_{count}_rank_trajectory')
        example_text.extend([f'## {targets[index]}',''])
        for method in methods:
            example_text.extend([f'### {LABELS.get(method,method)}','',
                '| Round | Rank | Question | Answer | Query |','|---|---:|---|---|---|'])
            for r in rounds:
                info=(details or {}).get((method,targets[index],int(r)),{})
                clean=lambda value:str(value or '').replace('|','/').replace('\n',' ')
                example_text.append(f'| {r} | {arrays[method][index,r]} | {clean(info.get("question"))} | {clean(info.get("answer"))} | {clean(info.get("query"))} |')
            example_text.append('')
    (destination.parent/'example_conversations.md').write_text('\n'.join(example_text)+'\n',encoding='utf-8')

    if 'agents_critique' in methods or 'agents_no_critique' in methods:
        fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
        ax.set_xlim(0,12);ax.set_ylim(0,4);ax.axis('off')
        boxes=[(.1,2.5,1.8,1,'Shared original\ncaption + asked QA'),(2.3,2.5,2,1,'Three role queries\n+ retrieval/clustering'),
               (4.7,2.5,2,1,'One question/agent\n+ shared filtering'),(7.1,2.5,2,1,'Optional peer critique\n+ one revision'),
               (9.5,2.5,2.3,1,'Common neutral KL\nselects ONE question'),(9.5,.5,2.3,1,'BLIP-2 answers\nusing target image'),
               (5.6,.5,3.3,1,'Neutral reformulation → official rank\nNo fusion in this experiment')]
        for x,y,w,h,label in boxes:
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',facecolor='white',edgecolor=colours[0]))
            ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
        for a,b in [((1.9,3),(2.3,3)),((4.3,3),(4.7,3)),((6.7,3),(7.1,3)),((9.1,3),(9.5,3)),((10.65,2.5),(10.65,1.5)),((9.5,1),(8.9,1)),((5.6,1),(1,2.5))]:
            ax.annotate('',b,a,arrowprops={'arrowstyle':'->','color':'#52514e'})
        ax.set_title('Our question-generation experiment',loc='left',fontweight='bold')
        save(fig,'fig1_setup')


def build_report(data, output, *, methods=None, comparisons=None, states=None, target_order=None,
                 allow_intersection=False, samples=5000, seed=42, figures=True, examples=3):
    """Write a new portable analysis folder; never overwrite source runs or silently drop targets."""
    if type(examples) is not int or examples<1:
        raise ValueError("Request at least one illustrative example.")
    arrays,targets,alignment=data.align(methods,states=states,target_order=target_order,allow_intersection=allow_intersection)
    names=list(arrays)
    reference=next((m for m in ('agents_no_critique','live_plugir_qwen','plugir','caption_only') if m in arrays),names[0])
    comparisons=[(m,reference) for m in names if m!=reference] if comparisons is None else comparisons
    result=statistics(arrays,comparisons,samples=samples,seed=seed)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    notes=["Only matched complete target trajectories are analyzed; exclusions are explicitly listed.",
        "Paired bootstrap resamples targets, keeping each target's rounds together; it does not measure model-seed variability.",
        "Accumulating-target bands are descriptive pointwise intervals, not a sequential stopping rule.",
        "A confidence interval including zero is inconclusive; it does not establish equivalence.",
        "Caption-mix/retrieval variants require real saved ranks; they are not invented by this reporter.",
        "Same graph format does not establish identical models, prompts, interaction budgets, candidate pools or causal comparability."]
    if len(targets)<2:
        notes.append("One-target pilot: confidence intervals unavailable; these plots are debugging output, not performance evidence.")
    report={**result,'protocol':{**alignment,'targets':targets,'methods':names,'comparisons':comparisons,
        'bootstrap_samples':samples,'seed':seed,'median':'lower middle order statistic, matching teammate torch.median',
        'bri':'mean trapezoidal natural log of one-based best-so-far ranks; undefined with one state',
        'sources':data.sources,'notes':notes}}
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_csv(output/'per_round.csv',result['per_round'])
    write_csv(output/'final_metrics.csv',result['final_metrics'])
    wide=[{'target':target,'method':method,**{f'round_{r}':int(arrays[method][i,r]) for r in range(arrays[method].shape[1])}}
          for i,target in enumerate(targets) for method in names]
    write_csv(output/'per_target_ranks.csv',wide)
    flat=[]
    for label,values in result['paired_differences'].items():
        for metric,interval in values.items():
            if metric!='final_rank_better_worse_same':
                flat.append({'comparison':label,'metric':metric,'difference':interval[0],'ci95_low':interval[1],'ci95_high':interval[2]})
    write_csv(output/'paired_differences.csv',flat)
    write_csv(output/'recall_hits_by_round.csv',[{'method':m,'metric':metric,
        **{f'round_{row["round"]}':row[metric] for row in result['per_round'] if row['method']==m}}
        for m in names for metric in ('recall@1','recall@10','hits@10')])
    (output/'REPORT_NOTES.md').write_text('# Reporting protocol\n\n'+'\n'.join('- '+note for note in notes)+'\n',encoding='utf-8')
    if figures:
        make_figures(arrays,targets,result,comparisons,output/'figures',seed=seed,examples=examples,details=data.details)
    return report


def main(argv=None):
    """CLI entry for existing reports/exports; analysis never downloads weights or starts a GPU."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',help='report.json, result folder, rank CSV, or METHOD=compact_conversations.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--methods',nargs='+')
    parser.add_argument('--compare',nargs=2,action='append',metavar=('LEFT','RIGHT'))
    parser.add_argument('--states',type=int)
    parser.add_argument('--target-order',help='JSON targets list, {targets:[...]}, or matched query list with img keys')
    parser.add_argument('--allow-intersection',action='store_true')
    parser.add_argument('--bootstrap',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--examples',type=int,default=3)
    parser.add_argument('--no-figures',action='store_true')
    args=parser.parse_args(argv)
    data=RankData()
    for source in args.inputs:
        data.load(source)
    order=None
    if args.target_order:
        order=json.loads(Path(args.target_order).read_text(encoding='utf-8-sig'))
        order=order['targets'] if isinstance(order,dict) else order
        order=[r['img'] if isinstance(r,dict) else r for r in order]
    result=build_report(data,args.output,methods=args.methods,comparisons=args.compare,states=args.states,
        target_order=order,allow_intersection=args.allow_intersection,samples=args.bootstrap,seed=args.seed,
        figures=not args.no_figures,examples=args.examples)
    print(f'Saved analysis: {Path(args.output).resolve()}')
    print(f'Matched targets: {len(result["protocol"]["targets"])}; states: {result["protocol"]["states"]}')
    for row in result['final_metrics']:
        print(f"{row['method']}: R@10={row['recall@10']:.1f}%, H@10={row['hits@10']:.1f}%, BRI={row['BRI']}")

