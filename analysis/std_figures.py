"""Figures at their final ICML print size; fixed canvases, reserved legend space.

No tight bounding-box cropping: a point in this source remains a point in the
paper. Reads results/ directly.
"""
from pathlib import Path
import json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.text import Text, Annotation
from matplotlib.colors import LinearSegmentedColormap
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from std_analyze import CHECKS, read, evreach, wilson  # noqa: E402
from _paths import OUT  # noqa: E402

FIG = Path(OUT)
D = json.loads((CHECKS/'analysis_std.json').read_text())
BLUE, RED, TEAL = '#2A78D6', '#E7683C', '#199D89'
C = [BLUE, RED, TEAL, '#9B78AB', '#A0803E']
INK, MUTED, GREY, GRID = '#25313B', '#65717B', '#919CA5', '#E7EBEE'
FROZEN = '#91B5D4'
CMAP = LinearSegmentedColormap.from_list('identity', ['#F4F7FA', '#B5D2E7', '#3179AF'])
plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':8,
    'axes.titlesize':9, 'axes.labelsize':8, 'axes.labelcolor':INK,
    'text.color':INK, 'axes.edgecolor':'#84909A', 'axes.linewidth':.6,
    'axes.spines.top':False, 'axes.spines.right':False,
    'xtick.labelsize':7.5, 'ytick.labelsize':7.5,
    'xtick.color':MUTED, 'ytick.color':MUTED,
    'xtick.major.width':.6, 'ytick.major.width':.6,
    'xtick.major.size':2.5, 'ytick.major.size':2.5,
    'xtick.major.pad':3, 'ytick.major.pad':3,
    'legend.fontsize':7.5, 'legend.frameon':False,
    'legend.handlelength':1.5, 'legend.handletextpad':.5,
    'legend.columnspacing':1, 'legend.labelspacing':.35,
    'lines.linewidth':1.35, 'lines.markersize':3.2,
    'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.dpi':240,
})

def canvas(height, width=6.75):
    return plt.figure(figsize=(width,height), facecolor='white')

def axes(fig,x,y,w,h):
    fw,fh=fig.get_size_inches()
    return fig.add_axes([x/fw,y/fh,w/fw,h/fh])

def text(fig,x,y,s,**kw):
    fw,fh=fig.get_size_inches()
    return fig.text(x/fw,y/fh,s,**kw)

def title(fig,x,y,letter,label):
    text(fig,x,y,letter,fontsize=9,weight='bold',va='baseline')
    text(fig,x+.19,y,label,fontsize=9,weight='bold',va='baseline')

def tidy(ax):
    ax.grid(axis='y',color=GRID,lw=.5,zorder=0)
    ax.set_axisbelow(True)

def legend(fig,ax,x,y,**kw):
    fw,fh=fig.get_size_inches()
    return fig.legend(*ax.get_legend_handles_labels(),loc='upper left',
                      bbox_to_anchor=(x/fw,y/fh),borderaxespad=0,**kw)

LAYOUT = {}

def save(fig,name):
    # A fixed canvas is essential: bbox_inches='tight' changed the effective
    # type size from one old figure to the next when LaTeX fit their widths.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    labels=[]; clipped=[]; collisions=[]
    skipped=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            ticks=axis.get_major_ticks()+axis.get_minor_ticks()
            lo,hi=sorted(ax.get_xlim() if axis is ax.xaxis else ax.get_ylim())
            for tick in ticks:
                if not ax.axison or not axis.get_visible() or not lo<=tick.get_loc()<=hi:
                    skipped.update([id(tick.label1),id(tick.label2)])
            if not ax.axison or not axis.get_visible():
                skipped.update([id(axis.label),id(axis.offsetText)])
    for t in fig.findobj(match=Text):
        if id(t) in skipped or not t.get_visible() or not t.get_text().strip():continue
        b=Text.get_window_extent(t,renderer) if isinstance(t,Annotation) else t.get_window_extent(renderer)
        # Tick objects outside the view can remain instantiated but are not drawn.
        if t.axes and t in t.axes.get_xticklabels()+t.axes.get_yticklabels():
            tx,ty=t.get_position()
            if t in t.axes.get_xticklabels() and not min(t.axes.get_xlim())<=tx<=max(t.axes.get_xlim()):continue
            if t in t.axes.get_yticklabels() and not min(t.axes.get_ylim())<=ty<=max(t.axes.get_ylim()):continue
        if b.width<=0 or b.height<=0:continue
        label=t.get_text()
        if b.x0 < -.5 or b.y0 < -.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:
            clipped.append(label)
        labels.append((label,b))
    for i,(a,ba) in enumerate(labels):
        for b,bb in labels[i+1:]:
            dx=min(ba.x1,bb.x1)-max(ba.x0,bb.x0)
            dy=min(ba.y1,bb.y1)-max(ba.y0,bb.y0)
            if dx>1 and dy>1:collisions.append([a,b])
    LAYOUT[name]={'canvas_inches':fig.get_size_inches().tolist(),'clipped_text':clipped,'overlapping_text':collisions}
    fig.savefig(FIG/(name+'.pdf'))
    fig.savefig(FIG/(name+'.png'))
    plt.close(fig)

# 1: First-page result, with a compact, explicit task definition.
fig=canvas(3.15,3.25)
title(fig,.04,3.00,'a','Follow a reference chain')
ax=axes(fig,.04,2.13,3.17,.75);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.add_patch(FancyBboxPatch((.005,.01),.99,.98,boxstyle='round,pad=0,rounding_size=.035',
                           facecolor='#F3F6F8',edgecolor='#DFE6EA',lw=.6))
for i,(a,b) in enumerate([('K = apple','M = pear'),('B = K','N = M'),('D = B','P = N')]):
    ax.text(.045,.80-i*.245,a,fontfamily='DejaVu Sans Mono',fontsize=8.5,color=BLUE,va='center')
    ax.text(.51,.80-i*.245,b,fontfamily='DejaVu Sans Mono',fontsize=8.5,color=MUTED,va='center')
text(fig,.12,1.97,'print(D)',fontfamily='DejaVu Sans Mono',fontsize=8.5)
text(fig,.93,1.97,'→',fontsize=10,color=MUTED)
text(fig,1.10,1.97,'apple',fontfamily='DejaVu Sans Mono',fontsize=9,weight='bold',color=RED)
text(fig,3.16,1.98,'one forward pass',ha='right',fontsize=7.5,color=MUTED)
title(fig,.04,1.75,'b','65K parameters unlock long chains')
ax=axes(fig,.54,.57,2.56,1.00)
M=read('e33_order_r8kl_matrix')['res'];L=read('e19_reentry_long_q8_r8_kl1')
base=[100*M['c2_forward_d24']['frozen']['acc'],100*L['eval']['d48_K0']]
mapped=[100*M['c2_forward_d24']['map']['acc'],100*L['eval']['d48_K1']]
x=np.array([0.,1.]);width=.28
for dx,ys,c,label in [(-.16,base,FROZEN,'Frozen'),(.16,mapped,RED,'+ rank-8 map')]:
    ax.bar(x+dx,ys,width=width,color=c,zorder=3,label=label)
    for xx,v in zip(x+dx,ys):
        ax.text(xx,v+4,f'{v:g}%',ha='center',va='bottom',fontsize=11 if c==RED else 8,
                weight='bold' if c==RED else 'normal',color=RED if c==RED else INK)
ax.set(ylim=(0,118),xlim=(-.55,1.55),yticks=[0,50,100],ylabel='Exact accuracy (%)',
       xticks=x,xticklabels=['24 lines','48 lines'])
ax.spines['left'].set_bounds(0,100);ax.tick_params(axis='x',length=0);tidy(ax)
# Conditions belong beside the bars, rather than being hidden in the caption.
for xx,label in zip(x,['train ≤20 lines','train ≤40 lines']):
    ax.text(xx,-.26,label,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=7.5,color=MUTED)
legend(fig,ax,.61,.16,ncol=2,fontsize=7.5)
save(fig,'fig_task')

# 2: V2's layer-by-position explanation, supported by causal traces below.
fig=canvas(4.25)
VIOLET='#6851A5'
W=read('e32b_wave_q8r8kl_d16')
selected=[1,2,3,4,8,12,16]
for i,mode in enumerate(['frozen','map']):
    x=.46 if i==0 else 3.96
    ax=axes(fig,x,2.58,2.62,1.36)
    title(fig,x-.16,4.08,'ab'[i],['Default: the relay stops','Map: the relay continues'][i])
    ax.set(xlim=(.55,8.85),ylim=(0,36),yticks=[0,14,24,32],
           xticks=list(range(1,8))+[8.4],xticklabels=['1','2','3','4','8','12','16','Q'],
           xlabel='Selected chain lines; Q = query',ylabel='Layer')
    ax.tick_params(axis='x',length=0)
    ax.spines['bottom'].set_visible(False)
    for xx in list(range(1,8))+[8.4]:ax.axvline(xx,color=GRID,lw=.45,zorder=0)
    ax.axhspan(30,34,color='#FDEEE5',zorder=0)
    ax.annotate('',(8.4,32),(1,32),arrowprops=dict(arrowstyle='->',color=RED,lw=1.2))
    ax.text(4.65,27.1,'Copy the selected value',ha='center',fontsize=7.5,color=RED)
    pts=[]
    for xx,k in enumerate(selected,1):
        if k==1:continue
        first=next((j for j,v in enumerate(W[mode]['grids']['rhs_slot'][k]) if v>=.75),None)
        if first is not None:pts.append((xx,first+1))
    ax.plot(*zip(*pts),c=BLUE,marker='o',ms=3.1,lw=1.5,zorder=3)
    if i==0:
        ax.scatter([4],[18],marker='x',s=25,c=RED,zorder=4)
        ax.text(1.1,21.4,'Relay stops',fontsize=7.5,color=BLUE)
        ax.annotate('',(7,20.2),(8.4,17),arrowprops=dict(arrowstyle='->',color=VIOLET,lw=1.2,connectionstyle='arc3,rad=.25'))
        ax.text(6.5,7,'Question resolves\nthe rest',fontsize=7.5,color=VIOLET,ha='center',linespacing=1.25)
    else:
        ax.axhline(14,c=TEAL,lw=.9,ls='--',zorder=1)
        ax.text(1,15.4,'Map',fontsize=7.5,color=TEAL)
        ax.text(4.6,5.3,'Relay continues',fontsize=7.5,color=BLUE,ha='center')
        ax.annotate('',(8.4,24),(7,24),arrowprops=dict(arrowstyle='->',color=VIOLET,lw=1.2,connectionstyle='arc3,rad=-.3'))
        ax.text(7.8,6.2,'Query reads\nthe relay',fontsize=7.5,color=VIOLET,ha='center',linespacing=1.25)

ax=axes(fig,.46,.75,2.62,1.09);title(fig,.30,2.03,'c','The value moves late')
for d,c in zip([1,2,3,4],['#9BBED8','#6E9FC3','#397DB5','#16456A']):
    v=read(f'e4_trace_q8base_val_d{d}')['agg']
    ax.plot(v['q|1|rhs'],color=c,label=f'{d} line'+('s' if d>1 else ''))
    ax.plot(v['query|4|:'],color=c,ls='--',lw=1.1)
ax.axvline(32,color=MUTED,ls=':',lw=.8)
ax.set(xlim=(10,36),ylim=(-.03,1.08),xlabel='Layer (input)',ylabel='Restored effect',yticks=[0,.5,1],xticks=[10,20,32]);tidy(ax)
legend(fig,ax,.77,.30,ncol=2,columnspacing=2.5)
ax=axes(fig,3.96,.75,2.62,1.09);title(fig,3.80,2.03,'d','Pointers are resolved earlier')
P=read('e21_adapted_q8_noloop_d5')['off']
for k,c in zip([5,4,3,2],[BLUE,TEAL,C[4],RED]):
    v=P[f'ptr{k}']['agg'];ax.plot(v[f'q|{k}|rhs'],color=c,label=f'Line {k}')
    ax.plot(v['query|last'],color=c,ls='--',lw=1.1)
ax.axvspan(17,24,color='#F1F3F5',zorder=0);ax.axvline(20.5,color=MUTED,ls=':',lw=.8)
ax.set(xlim=(10,30),ylim=(-.03,1.08),xlabel='Layer (input)',ylabel='Restored effect',yticks=[0,.5,1],xticks=[10,20,30]);tidy(ax)
legend(fig,ax,4.27,.30,ncol=2,columnspacing=2.5)
save(fig,'fig_default_std')

# 3: Restore V2's paired relay heatmaps and follow the causal pointer effect.
fig=canvas(2.83)
xs=[.48,2.03,3.58,5.32];ws=[1.27,1.27,1.27,1.22]
axs=[axes(fig,x,.88,w,1.59) for x,w in zip(xs,ws)]
for x,l,t in zip(xs,'abcd',['Default','With the map','Effect moves on','Attention reach']):title(fig,x-.12,2.66,l,t)
for i,mode in enumerate(['frozen','map']):
    ax=axs[i];grid=np.array(W[mode]['grids']['rhs_slot'])[2:]
    fin=np.array(W[mode]['grids']['final_slot'])[None]
    heat=np.vstack([grid,np.full((1,36),np.nan),fin])
    im=ax.imshow(heat,origin='lower',aspect='auto',cmap=CMAP,vmin=.5,vmax=1,
                 interpolation='nearest',extent=(-.5,35.5,1.5,18.5))
    front=[]
    for k in range(2,17):
        j=next((j for j,v in enumerate(W[mode]['grids']['rhs_slot'][k]) if v>=.75),None)
        if j is not None:front.append((j,k))
    ax.plot(*zip(*front),c=RED,lw=1.0,marker='o',ms=2.0)
    q=next((j for j,v in enumerate(fin[0]) if v>=.75),None)
    if q is not None:ax.scatter(q,18,c=RED,s=10)
    ax.set(xlabel='Layer (output)',xticks=[0,14,28],yticks=[2,4,8,12,16,18])
    if i==0:ax.set_yticklabels(['2','4','8','12','16','Q']);ax.set_ylabel('Line in chain; Q = query')
    else:ax.set_yticklabels([])
    ax.tick_params(axis='y',length=2 if i==0 else 0)
    if mode=='map':ax.axvline(13.5,color=TEAL,lw=.85,ls='--')
cax=axes(fig,.69,.33,2.4,.08);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_ticks([.5,.75,1])
text(fig,1.89,.055,'Chain-identity accuracy',fontsize=7.5,ha='center')

ax=axs[2];flow=read('e34_deep_trace_q8r8kl_d16')['map']['K2']['agg']
heat=np.array([flow[f'q|{k}|rhs'] for k in range(2,17)]+[[np.nan]*37,flow['query|last']])
orange=LinearSegmentedColormap.from_list('pointer',['#FFFFFF','#F8CCAC','#E7683C','#963616'])
im=ax.imshow(heat,origin='lower',aspect='auto',cmap=orange,vmin=0,vmax=1,
             interpolation='nearest',extent=(-.5,36.5,1.5,18.5))
ax.set(xlabel='Layer (input)',xticks=[0,18,36],yticks=[])
cax=axes(fig,xs[2],.33,ws[2],.08);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_ticks([0,.5,1])
text(fig,xs[2]+ws[2]/2,.055,'Restored effect',fontsize=7.5,ha='center')
ax=axs[3];S=np.array(read('e37b_strides_q8r8kl_d16_inter')['map']['diff']).T
im=ax.imshow(S[:,14:27],origin='lower',aspect='auto',cmap=CMAP,vmin=0,vmax=.5,extent=(13.5,26.5,.5,S.shape[0]+.5))
ax.set(xlabel='Layer',ylabel='Lines up the chain',xticks=[14,20,26],yticks=[1,4,8,12])
cax=axes(fig,xs[3],.33,ws[3],.08);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_ticks([0,.25,.5])
text(fig,xs[3]+ws[3]/2,.055,'Same − other',fontsize=7.5,ha='center')
save(fig,'fig_relay_std')

# 4: Shared legend outside curves; direct labels with reserved space.
fig=canvas(2.93);ax=axes(fig,.45,.89,2.66,1.70)
title(fig,.29,2.76,'a','The map works only up to a layer')
for i,(name,v) in enumerate(D['placement'].items()):
    if v['kind']!='development':continue
    pts=sorted((int(a),r) for a,r in v['reach'].items());x,y=zip(*pts)
    ax.plot(np.array(x)/v['N'],y,c=C[i],marker='o',ms=3,label=name)
ax.set(xlabel='Map layer / number of layers',ylabel='Reach (lines)',ylim=(0,26),xlim=(0,.96),yticks=[0,8,16,24],xticks=[0,.3,.6,.9]);tidy(ax)
legend(fig,ax,.48,.51,ncol=2)
ax=axes(fig,3.95,.89,2.61,1.70);title(fig,3.79,2.76,'b','The frozen model predicts that layer')
labels={'Llama-3.2-1B':('Llama 1B',10,7), 'Qwen3-4B':('Qwen 4B',25.2,18.7),
        'Gemma-3-12B':('Gemma 12B',25.0,27.8), 'OLMo-3-32B':('OLMo 32B',10.8,27.6)}
for name,v in D['placement'].items():
    x=v['predictions']['commit'];y=v['midpoint'];lo,hi=v['bracket'];held=v['kind']=='held-out'
    ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt='o',ms=4,c=RED if held else GREY,
                mfc=RED if held else 'white',capsize=3,lw=1)
    if held:
        label,tx,ty=labels[name]
        ax.annotate(label,(x,y),xytext=(tx,ty),fontsize=7.5,ha='left',va='center',
                    arrowprops=dict(arrowstyle='-',color='#A7AFB5',lw=.6,shrinkA=2,shrinkB=4),
                    bbox=dict(fc='white',ec='none',pad=.25))
ax.plot([5,31],[5,31],c='#B2BAC1',ls=':',lw=1,zorder=0)
ax.set(xlim=(5,34),ylim=(4.5,31),xlabel='Cutoff layer (frozen model)',ylabel='Last layer where the map works',xticks=[8,16,24,32],yticks=[8,16,24]);tidy(ax)
handles=[Line2D([],[],color=GREY,marker='o',mfc='white',ls='',label='Development'),Line2D([],[],color=RED,marker='o',ls='',label='Held out')]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(3.98/6.75,.40/2.93),ncol=2,borderaxespad=0)
text(fig,5.25,.09,'Bars: last working and first failing layer tested',ha='center',fontsize=7.5,color=MUTED)
save(fig,'fig_placement')

# 5: MuSiQue with fixed prompts (std_musique_paired.py): gains over the frozen model with paired 95% intervals over questions.
Q=json.loads((CHECKS/'musique_std.json').read_text())['models']
fig=canvas(2.90);ax=axes(fig,.45,.71,2.46,1.82)
title(fig,.29,2.73,'a','Early maps improve MuSiQue')
for tag,N,name,col,edge in [('q8',36,'Qwen 8B',BLUE,20.5),('o3',32,'OLMo 7B',RED,13.5),('l31',32,'Llama 8B',TEAL,14)]:
    if tag not in Q:continue
    pts=sorted((int(k.split('@')[1]),v) for k,v in Q[tag]['arms'].items() if k.startswith('map@'))
    xs=np.array([a for a,_ in pts])/N
    ax.plot(xs,[v['gain'] for _,v in pts],c=col,marker='o',ms=3,label=name,zorder=3)
    for x,(_,v) in zip(xs,pts):ax.plot([x,x],v['ci'],c=col,lw=.8,alpha=.6)
    ax.axvline(edge/N,c=col,lw=.8,ls=':',alpha=.75)
ax.axhline(0,c=GREY,lw=.6);ax.set(xlabel='Map layer / number of layers',ylabel='EM gain (points)',xlim=(0,.95),ylim=(-8,23),xticks=[0,.3,.6,.9],yticks=[-5,0,5,10,15,20]);tidy(ax)
legend(fig,ax,.41,.23,ncol=3,handlelength=1.0,columnspacing=.7)
ax=axes(fig,4.57,.71,2.02,1.82);title(fig,3.43,2.73,'b','LoRA: early, all, or late layers')
groups=[('Qwen3-8B','q8',['LoRA 0-20','LoRA all','LoRA 21-35'],'0–20 | 21–35'),('OLMo-3-7B','o3',['LoRA 0-14','LoRA all','LoRA 15-31'],'0–14 | 15–31'),
        ('Llama-3.1-8B','l31',['LoRA 0-14','LoRA all','LoRA 15-31'],'0–14 | 15–31')]
for i,(name,tag,arms,split) in enumerate(groups):
    if tag not in Q:continue
    for j,(arm,c,m) in enumerate(zip(arms,[BLUE,INK,RED],['o','D','s'])):
        v=Q[tag]['arms'].get(arm)
        if v is None:continue
        y=2-i+(1-j)*.17
        ax.plot(v['ci'],[y,y],c=c,lw=1.3)
        ax.scatter(v['gain'],y,c=c,marker=m,s=23,label=['Early','All','Late'][j] if i==0 else None,zorder=3)
    ax.text(-.09,2-i+.07,name,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8,clip_on=False)
    ax.text(-.09,2-i-.15,split,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=7.5,color=MUTED,clip_on=False)
ax.axvline(0,c=GREY,lw=.7);ax.set(xlim=(-4,26),ylim=(-.5,2.5),yticks=[],xticks=[0,10,20],xlabel='EM gain (points)')
for y in [.5,1.5]:ax.axhline(y,color=GRID,lw=.5,zorder=0)
ax.spines['left'].set_visible(False)
legend(fig,ax,4.55,.23,ncol=3,handlelength=1,columnspacing=.8)
save(fig,'fig_qa_std')

# A1: Four fully labelled sweeps; integer ticks and one shared legend.
fig=canvas(4.25)
for i,(name,v) in enumerate((n,v) for n,v in D['placement'].items() if v['kind']=='held-out'):
    x=.47+(i%2)*3.39;y=2.59 if i<2 else .79
    ax=axes(fig,x,y,2.69,1.16);title(fig,x-.16,y+1.34,'abcd'[i],name)
    xs=sorted(map(int,v['reach']));ys=[v['reach'][str(t)] for t in xs]
    ax.plot(xs,ys,c=BLUE,marker='o',ms=3,label='Seed 0')
    pts=[(int(a),ss['1']) for a,ss in v['seeds'].items() if '1' in ss]
    if pts:ax.scatter(*zip(*pts),marker='x',s=25,c=RED,label='Seed 1',zorder=4)
    ax.axhline(v['threshold'],c=GREY,ls='--',lw=.8,label='Working threshold')
    ax.axvline(v['predictions']['commit'],c=TEAL,ls=':',lw=1.2,label='Cutoff-layer prediction')
    ax.axvspan(*v['bracket'],color='#E9F1F7',zorder=0)
    xticks=[1,4,8,12] if '1B' in name else ([12,16,20,24,26] if 'Qwen' in name else [14,21,27,34])
    ax.set(ylabel='Reach (lines)',ylim=(0,8 if '1B' in name else 26),xticks=xticks,
           yticks=[0,4,8] if '1B' in name else [0,8,16,24]);tidy(ax)
    if i==0:lh=ax.get_legend_handles_labels()
text(fig,3.48,.37,'Map layer',fontsize=8,ha='center')
fig.legend(*lh,loc='lower center',bbox_to_anchor=(.5,.015),ncol=4,borderaxespad=0,columnspacing=1.5)
save(fig,'fig_heldout')

# A2: Causal evidence, pretraining, and transfer, in the same visual grammar.
fig=canvas(2.92);xs=[.47,2.74,5.01];w=1.57
axs=[axes(fig,x,1.00,w,1.55) for x in xs]
for x,l,t in zip(xs,'abc',['Both head sets matter','Acquired in pretraining','Transfer vs. distance']):title(fig,x-.16,2.74,l,t)
ax=axs[0];AB=read('e66_ablate_q8r8kl_inter_after');ab=AB['acc'];ds=[8,12,16]
for key,label,c,ls in [('none','Intact',BLUE,'-'),('short','Parent',RED,'-'),('random_short',f"Random {len(AB['rand_short'])}",RED,'--'),('long','Further up',TEAL,'-'),('random',f"Random {len(AB['rand_heads'])}",TEAL,'--')]:
    ax.plot(ds,[ab[key][str(d)] for d in ds],c=c,ls=ls,marker='o',ms=2.8,label=label)
ax.set(xlabel='Chain length (lines)',ylabel='Choice accuracy',ylim=(.4,1.04),xticks=ds,yticks=[.4,.6,.8,1]);tidy(ax)
legend(fig,ax,xs[0]-.02,.49,ncol=2,handlelength=1,columnspacing=.6)
ax=axs[1]
for a,c,label in [(6,RED,'Map at layer 6'),(12,TEAL,'Map at layer 12')]:
    ys=[]
    for step in [5000,20000,80000]:
        suffix='_a6_r8' if a==6 else ''
        ys.append(evreach(read(f'e19_reentry_ckpt_olmo_stage1step{step}'+suffix)))
    ax.plot([21,84,336],ys,c=c,marker='o',ms=3,label=label)
base=[evreach(read(f'e19_reentry_ckpt_olmo_stage1step{s}'),0) for s in [5000,20000,80000]]
ax.plot([21,84,336],base,c=FROZEN,ls='--',marker='o',ms=3,label='Frozen')
ax.set(xscale='log',xticks=[21,84,336],xticklabels=['21','84','336'],xlabel='Pretraining tokens (B)',ylabel='Reach (lines)',ylim=(0,26),yticks=[0,8,16,24]);ax.minorticks_off();tidy(ax)
legend(fig,ax,xs[1]-.02,.49)
ax=axs[2];z=read('zoo_parents')
refs={'Qwen/Qwen3-1.7B-Base','Qwen/Qwen3-4B-Base','Qwen/Qwen3-8B-Base','allenai/Olmo-3-1025-7B','NousResearch/Meta-Llama-3.1-8B','allenai/Olmo-3-7B-Think-SFT','allenai/Olmo-3-7B-Think-DPO','allenai/Olmo-3-7B-Instruct'}
for name,v in z.items():
    if name in refs or 'd_map' not in v or v['dist_parent']<=0:continue
    ax.scatter(100*v['dist_parent'],v['d_map'],c=BLUE,s=19,alpha=.75,edgecolors='white',linewidths=.3)
ax.axhline(0,c=GREY,lw=.7);ax.set(xscale='log',xlabel='Weight distance (%)',ylabel='Change in reach (lines)',yticks=[-6,-4,-2,0,2]);ax.minorticks_off();tidy(ax)
text(fig,xs[2]+w/2,.36,'Same base-trained map;\nreference-relative change',ha='center',fontsize=7.5,color=MUTED,linespacing=1.4)
save(fig,'fig_controls_std')

# A3: Matched full accuracy curves, one panel per map-training setting.
fig=canvas(2.55)
for i,(train,kind) in enumerate([(20,'standard'),(40,'long')]):
    x=.47+i*3.46;ax=axes(fig,x,.70,2.59,1.48)
    title(fig,x-.16,2.37,'ab'[i],f'Map trained to {train} lines')
    if kind=='standard':
        ds=[2,4,6,8,12,16,20,24]
        yy=[M[f'c2_forward_d{d}']['map']['acc'] for d in ds]
        bb=[M[f'c2_forward_d{d}']['frozen']['acc'] for d in ds]
        n=200
    else:
        ev=read('e19_reentry_long_q8_r8_kl1')['eval']
        ds=sorted({int(k.split('_')[0][1:]) for k in ev})
        yy=[ev[f'd{d}_K1'] for d in ds];bb=[ev[f'd{d}_K0'] for d in ds];n=100
    ax.axvspan(train,max(ds)+1,color='#F5F0E9',zorder=0)
    ax.plot(ds,yy,c=RED,marker='o',ms=3,label='Rank-8 map')
    ax.plot(ds,bb,c=GREY,ls='--',marker='s',ms=2.5,label='Frozen')
    for d,y in zip(ds,yy):
        lo,hi=wilson(y,n);ax.errorbar(d,y,yerr=[[y-lo],[hi-y]],color=RED,lw=.6,capsize=1.6)
    ax.axhline(.8,color=GREY,lw=.6,ls=':')
    ax.set(ylim=(-.02,1.08),xlabel='Chain length (lines)',ylabel='Exact accuracy',
           xticks=[2,8,16,24] if i==0 else [2,16,32,48,64],yticks=[0,.5,1]);tidy(ax)
    legend(fig,ax,x+.18,.23,ncol=2)
save(fig,'fig_accuracy_std')

from std_analyze import USED
(CHECKS/'sources_std_figures.json').write_text(json.dumps(sorted(USED),indent=2))
(CHECKS/'figure_layout_std.json').write_text(json.dumps(LAYOUT,indent=2))
print('Generated 8 figures at exact ICML print widths (3.25 / 6.75 inches).')
