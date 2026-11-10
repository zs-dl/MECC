"""Plot portable, aggregate evidence CSVs; this script never runs an estimator."""
import csv,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'
HERE=ROOT/'outputs/figures'
HERE.mkdir(parents=True,exist_ok=True)
WIDTH=516/72.27
INPUTS=['Zipf-128','Zipf-1024','Zipf-4096','MovieLens','Kosarak','Uniform']
GROUPS=['rank_1_32','rank_33_128','rank_129_plus','absent']
LABELS=['1–32','33–128','129+','Absent']
BLUE='#27698d';RED='#b44332'
try:
 FONT=font_manager.findfont('Times New Roman',fallback_to_default=False); FAMILY='Times New Roman'
except ValueError:
 FONT=font_manager.findfont('DejaVu Serif'); FAMILY='DejaVu Serif'
plt.rcParams.update({'font.family':FAMILY,'font.size':9.5,'axes.labelsize':9.5,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'savefig.dpi':300})
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name):
 with (DATA/name).open() as f:return list(csv.DictReader(f))
manifest={'script_sha256':sha(__file__),'width_TeXpt':516,'dpi':300,'font_family':FAMILY,'font_sha256':sha(FONT),'sources':{name:sha(DATA/name) for name in ['coordinate-gains.csv','row-alignment.csv','row-alignment-statistics.csv']},'figures':{}}
def clean(ax,index):
 ax.spines[['top','right']].set_visible(False);ax.tick_params(length=2.5,pad=2);ax.text(0,1.09,f'({chr(97+index)})  {INPUTS[index]}',transform=ax.transAxes,ha='left',va='bottom',fontsize=9.5)
def save(fig,name,description):
 fig.canvas.draw()
 for ext in ['pdf','png']:fig.savefig(HERE/f'{name}.{ext}',dpi=300,facecolor='white')
 manifest['figures'][name]={'width_inches':fig.get_figwidth(),'height_inches':fig.get_figheight(),'description':description,'files':{f'{name}.{e}':sha(HERE/f'{name}.{e}') for e in ['pdf','png']}}
 plt.close(fig)
def coordinate():
 rr=read('coordinate-gains.csv');L={(r['input'],r['stage'],r['frequency_group']):r for r in rr}
 fig,axes=plt.subplots(2,3,figsize=(WIDTH,3.15));fig.subplots_adjust(left=.082,right=.988,bottom=.16,top=.80,wspace=.37,hspace=.60)
 handles=[]
 for i,(name,ax) in enumerate(zip(INPUTS,axes.ravel())):
  limits=[0.]
  for stage,color,marker,offset,label in [('pre',BLUE,'o',.13,'Before projection'),('post',RED,'s',-.13,'After projection')]:
   for j,g in enumerate(GROUPS):
    r=L[name,stage,g]
    if int(r['coordinates'])==0:continue
    v,lo,hi=map(float,[r['mean_pp'],r['ci95_low_pp'],r['ci95_high_pp']]);limits.extend([lo,hi])
    h=ax.errorbar(v,3-j+offset,xerr=[[v-lo],[hi-v]],fmt=marker,color=color,ms=3.1,lw=.8,capsize=1.6,zorder=4,label=label)
    if i==0 and j==0:handles.append(h)
  lo,hi=min(limits),max(limits);pad=max((hi-lo)*.10,.0001);ax.set_xlim(lo-pad,hi+pad)
  for j,g in enumerate(GROUPS):
   if int(L[name,'pre',g]['coordinates'])==0:ax.text(.5,3-j,'n/a',transform=ax.get_yaxis_transform(),ha='center',va='center',color='#727c83',fontsize=9)
  ax.axvline(0,color='#7d858b',lw=.65);ax.set_ylim(-.45,3.45);ax.set_yticks(range(4),LABELS[::-1]);ax.xaxis.set_major_locator(MaxNLocator(nbins=3));ax.grid(axis='x',color='#e3e7e9',lw=.5);clean(ax,i)
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.997),ncol=2,frameon=False,columnspacing=1.8,handlelength=2.2)
 fig.text(.52,.025,'Contribution to full-domain MSE reduction (pp)',ha='center',fontsize=9.5)
 save(fig,'coordinate-gains','All six inputs at epsilon=5, 100 fixed-input randomizations. Four true empirical-frequency strata; ties by category ID. Both stages use the same mean POST-projection Calibrate-public full-domain MSE denominator. Positive contributions are paired squared-error reductions, divided by full d. Unadjusted paired t95 intervals; n/a means empty group. Independent linear x-axis limits per panel.')
def alignment():
 rows=read('row-alignment.csv');fig,axes=plt.subplots(2,3,figsize=(WIDTH,3.05));fig.subplots_adjust(left=.09,right=.988,bottom=.13,top=.82,wspace=.37,hspace=.60)
 for i,(name,ax) in enumerate(zip(INPUTS,axes.ravel())):
  rr=sorted([r for r in rows if r['input']==name],key=lambda r:int(r['seed']));assert len(rr)==10
  vals=np.array([[float(r['aligned_change_pp']),float(r['shuffled_change_pp'])] for r in rr])
  for k,v in enumerate(vals):
   jitter=(k-4.5)*.006;ax.plot([jitter,1+jitter],v,color='#a0aab0',lw=.65,alpha=.85,zorder=2);ax.plot(jitter,v[0],'o',color=BLUE,ms=2.8,zorder=3);ax.plot(1+jitter,v[1],'s',color=RED,ms=2.8,zorder=3)
  ax.plot([0,1],vals.mean(0),ls='none',marker='D',ms=4.2,mfc='white',mec='#192f39',mew=1,zorder=5,label='Mean')
  ax.axhline(0,color='#7d858b',lw=.65);lo=min(0.,float(vals.min()));hi=max(0.,float(vals.max()));pad=max((hi-lo)*.10,.0001);ax.set_ylim(lo-pad,hi+pad);ax.set_xlim(-.22,1.22)
  ax.set_xticks([0,1],['Aligned','Shuffled']);ax.yaxis.set_major_locator(MaxNLocator(nbins=3));ax.grid(axis='y',color='#e3e7e9',lw=.5);clean(ax,i)
 fig.text(.018,.51,'Excess MSE over base (pp)',ha='center',va='center',rotation=90,fontsize=9.5)
 handles=[plt.Line2D([0],[0],marker='o',color=BLUE,lw=0,ms=3.3,label='Aligned'),plt.Line2D([0],[0],marker='s',color=RED,lw=0,ms=3.3,label='Shuffled'),plt.Line2D([0],[0],marker='D',mfc='white',mec='#192f39',color='#192f39',lw=0,ms=4,label='Mean')]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.997),ncol=3,frameon=False,columnspacing=1.8,handletextpad=.5)
 save(fig,'row-alignment','All six inputs at epsilon=5, all ten paired report realizations per input. Each thin line connects one original report and its row-shuffled counterfactual. Points are 100*(variant MSE - same-seed Calibrate-public MSE)/mean original Calibrate-public MSE. Positive means excess error. Hollow diamonds are means; no confidence intervals drawn. Independent linear y-axis limits retain all observations. Paired-shift t95 intervals provided separately in CSV.')
coordinate();alignment();(HERE/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'figures':list(manifest['figures']),'total_height_inches':6.2}))
