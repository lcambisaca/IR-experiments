from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

output = Path(__file__).parent
rounds = np.arange(11)
datasets = {
    'Cake': {
        'Live PlugIR': [9,21,1,2,1,1,1,1,1,1,1],
        'Critique': [9,17,21,10,15,8,8,16,9,1,1],
        'Debate': [9,1,5,2,2,2,2,2,2,1,1],
    },
    'Giraffe': {
        'Live PlugIR': [89,50,98,120,260,155,194,150,124,114,127],
        'Critique': [89,68,109,110,86,76,63,43,87,134,87],
        'Debate': [89,172,151,160,149,149,149,145,171,318,165],
    },
}
styles = {'Live PlugIR': ('#2563a6','o'), 'Critique': ('#e28c29','s'), 'Debate': ('#16856b','^')}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, layout='constrained')
for ax, (target, methods) in zip(axes, datasets.items()):
    upper = 25 if target == 'Cake' else 150
    ax.axhspan(1,10,color='#dbeee4',alpha=.8,zorder=0)
    ax.axhline(10, color='#657568', linestyle='--', linewidth=1, label='Top 10')
    for method, values in methods.items():
        color, marker = styles[method]
        vals = np.asarray(values)
        ax.plot(rounds, np.minimum(vals,upper), marker=marker, color=color, linewidth=2.25, markersize=6, label=method)
        for r in rounds[vals>upper]:
            ax.scatter(r, upper, marker='v', color=color, s=50, clip_on=False, zorder=5)
            ax.annotate(str(vals[r]), (r,upper), xytext=(0,7 if method=='Debate' else 18), textcoords='offset points', ha='center', va='top', color=color, fontsize=8, fontweight='bold',clip_on=False)
    ax.set_title(target, fontsize=17,fontweight='bold',loc='left')
    ax.set_xticks(rounds)
    ax.set_xlabel('Interaction state (0 = initial caption)')
    ax.set_ylim(upper,0)
    ax.set_ylabel('Target rank (lower is better)')
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.set_yticks([1,5,10,15,20,25] if target=='Cake' else [1,10,25,50,75,100,125,150])
axes[1].legend(loc='upper right', frameon=True, fontsize=9)
fig.suptitle('Qwen-VL pilot: retrieval rank by interaction state', fontsize=19,fontweight='bold')
fig.supxlabel('Giraffe ranks >150 are clipped at the boundary and labeled with their exact value.', fontsize=10)
fig.savefig(output/'pilot_rank_by_round.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig)
