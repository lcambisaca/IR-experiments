from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

rounds = np.arange(11)
datasets = {
    'Cake': {
        'Live PlugIR': [9,21,1,2,1,1,1,1,1,1,1],
        'Critique': [9,17,21,10,15,8,8,16,9,1,1],
        'Debate v7': [9,1,5,2,2,2,2,2,2,1,1],
        'Debate v8': [9,7,7,10,3,2,2,2,1,1,1],
    },
    'Giraffe': {
        'Live PlugIR': [89,50,98,120,260,155,194,150,124,114,127],
        'Critique': [89,68,109,110,86,76,63,43,87,134,87],
        'Debate v7': [89,172,151,160,149,149,149,145,171,318,165],
        'Debate v8': [89,88,28,54,40,31,69,48,93,93,93],
    },
}
styles = {
    'Live PlugIR': ('#2563a6','o','-',2),
    'Critique': ('#e28c29','s','-',2),
    'Debate v7': ('#16856b','^','--',2),
    'Debate v8': ('#8026b3','D','-',3),
}
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharex=True, layout='constrained')
for ax, (target, methods) in zip(axes, datasets.items()):
    upper = 25 if target == 'Cake' else 150
    ax.axhspan(1,10,color='#dbeee4',alpha=.8,zorder=0)
    ax.axhline(10, color='#657568', linestyle=':', linewidth=1.2, label='Top 10')
    for method, values in methods.items():
        color, marker, line, width = styles[method]
        vals = np.asarray(values)
        ax.plot(rounds, np.minimum(vals,upper), marker=marker, color=color, linestyle=line, linewidth=width, markersize=6, label=method, zorder=4 if method=='Debate v8' else 3)
        for r in rounds[vals>upper]:
            ax.scatter(r, upper, marker='v', color=color, s=40, clip_on=False,zorder=5)
            ax.annotate(str(vals[r]), (r,upper), xytext=(0,9 if method=='Debate v7' else 20), textcoords='offset points', ha='center', color=color, fontsize=8, fontweight='bold')
    if target=='Giraffe':
        ax.annotate('v8 best: 28', (2,28), xytext=(8,10), textcoords='offset points', color=styles['Debate v8'][0],fontweight='bold', fontsize=10)
    ax.set_title(target, fontsize=17,fontweight='bold',loc='left')
    ax.set_xticks(rounds)
    ax.set_xlabel('Interaction state (0 = initial caption)')
    ax.set_ylim(upper,0)
    ax.set_ylabel('Target rank (lower is better)')
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.set_yticks([1,5,10,15,20,25] if target=='Cake' else [1,10,25,50,75,100,125,150])
axes[1].legend(loc='upper right',frameon=True,fontsize=9)
fig.suptitle('Qwen-VL pilot: debate v8 versus v7 and baselines', fontsize=19,fontweight='bold')
fig.supxlabel('Giraffe ranks above 150 are clipped at the boundary and labeled with their exact values.', fontsize=10)
fig.savefig(Path(__file__).with_name('pilot_rank_v8_comparison.png'), dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig)
