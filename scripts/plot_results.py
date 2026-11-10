"""Regenerate manuscript figures from the bundled final summary statistics."""
import csv, hashlib, json, runpy
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/figures'
WIDTH=516/72.27; SINGLE=252/72.27
SCENARIOS=['zipf-d128-n10000','zipf-d1024-n10000','zipf-d4096-n10000','movielens-1m-first-movie','fimi-kosarak-click-d41270','uniform-d22000-n1024']
LABELS=['Zipf-128','Zipf-1024','Zipf-4096','MovieLens','Kosarak','Uniform']
EPS=[.5,1.,2.,3.,4.,5.]
BLUE='#27698d'; RED='#b44332'; GOLD='#b48a36'; GREEN='#27766b'; PURPLE='#80689e'
try:
 FONT=font_manager.findfont('Times New Roman',fallback_to_default=False); FAMILY='Times New Roman'
except ValueError:
 FONT=font_manager.findfont('DejaVu Serif'); FAMILY='DejaVu Serif'
plt.rcParams.update({'font.family':FAMILY,'font.size':9.5,'axes.labelsize':9.5,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'savefig.dpi':300})
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
SRC=ROOT/'results'
SUMMARY={(r['scenario'],float(r['epsilon']),r['method']):r for r in read(SRC/'summary.csv')}
PAIRED={(r['scenario'],float(r['epsilon'])):r for r in read(SRC/'paired.csv') if r['control']=='calibrate_pl0_public' and r['metric']=='mse'}
RISK={(r['scenario'],float(r['epsilon'])):r for r in read(SRC/'mechanism-summary.csv')}
VALUES=[]
MANIFEST={'script_sha256':sha(__file__),'font':{'family':FAMILY,'sha256':sha(FONT)},'dpi':300,'template_dimensions_TeXpt':{'columnwidth':252,'textwidth':516},'exact_width_export':True,'figures':{},'sources':{str(p.relative_to(ROOT)):sha(p) for p in [SRC/'summary.csv',SRC/'paired.csv',SRC/'mechanism-summary.csv']},'scientific_conditions':{'scenarios':dict(zip(SCENARIOS,LABELS)),'epsilons':EPS,'repetitions':100,'independent_unit':'OUE randomization on fixed empirical inputs','paired_intervals':'Unadjusted paired t 95% intervals, not ratio intervals; endpoints normalized by the observed control mean','envelope':'Minimum native baseline mean per condition; not one deployable method'}}
def save(fig,name,description):
 fig.canvas.draw()
 bounds=[]
 for t in fig.findobj(matplotlib.text.Text):
  if t.get_visible() and t.get_text():
   b=t.get_window_extent(fig.canvas.get_renderer()); bounds.append({'text':t.get_text(),'fontsize':t.get_fontsize(),'bounds_px':[b.x0,b.y0,b.x1,b.y1]})
 for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white')
 MANIFEST['figures'][name]={'width_inches':fig.get_figwidth(),'height_inches':fig.get_figheight(),'description':description,'files':{f'{name}.{e}':sha(OUT/f'{name}.{e}') for e in ['png','pdf']},'visible_text':bounds,'canvas_check_dpi':fig.dpi}
 plt.close(fig)
def mse():
 fig,axes=plt.subplots(2,3,figsize=(WIDTH,3.25));fig.subplots_adjust(left=.073,right=.991,bottom=.12,top=.80,wspace=.34,hspace=.48)
 spec=[('mecc','MECC',RED,'o','-'),('calibrate_pl0_public','Calibrate-public',BLUE,'s','--'),('calibrate_pl0_mle','Calibrate-MLE',GOLD,'^',':'),('oue_full_mle','Full-report MLE',PURPLE,'D','-.'),('native_envelope','Native envelope','#555c63','x',':')]
 handles=[]
 for i,(ax,s) in enumerate(zip(axes.ravel(),SCENARIOS)):
  for m,label,c,mark,ls in spec:
   yy=[float(SUMMARY[s,e,'mecc']['best_native_mse']) if m=='native_envelope' else float(SUMMARY[s,e,m]['mse_mean']) for e in EPS]
   line,=ax.plot(EPS,yy,color=c,marker=mark,ls=ls,lw=1.1,ms=3.0,label=label,zorder=5 if m=='mecc' else 3)
   if i==0:handles.append(line)
   VALUES.extend(dict(figure='mse',scenario=s,epsilon=e,method=m,value=v) for e,v in zip(EPS,yy))
  ax.set_yscale('log');ax.yaxis.set_major_locator(LogLocator(base=10,numticks=3));ax.yaxis.set_minor_formatter(NullFormatter());ax.set_xticks(EPS);ax.set_xticklabels(['0.5','1','2','3','4','5']);ax.tick_params(pad=1,length=2.5)
  ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',lw=.5,color='#e2e5e7');ax.text(.03,1.07,f'({chr(97+i)})',transform=ax.transAxes,ha='left');ax.text(.17,1.07,LABELS[i],transform=ax.transAxes,ha='left')
  if i>=3:ax.set_xlabel(r'$\epsilon$',labelpad=1)
  if i%3==0:ax.set_ylabel('MSE',labelpad=1)
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.996),ncol=3,frameon=False,columnspacing=1.15,handlelength=2.0,handletextpad=.5)
 save(fig,'mse','Six scenarios; logarithmic mean MSE from 100 repetitions. Five series retain proposed method, public-prior control, learned-prior Calibrate-MLE, full-report MLE, and the native baseline envelope. Other estimators remain in source CSVs.')
def heatmap():
 fig=plt.figure(figsize=(SINGLE,2.60));ax=fig.add_axes([.275,.305,.70,.65]);data=np.zeros((6,6));stars=np.zeros((6,6),bool)
 for i,s in enumerate(SCENARIOS):
  for j,e in enumerate(EPS):
   r=PAIRED[s,e];base=float(SUMMARY[s,e,'calibrate_pl0_public']['mse_mean']);v=-100*float(r['mecc_minus_control'])/base;lo=-100*float(r['ci95_high'])/base;hi=-100*float(r['ci95_low'])/base
   data[i,j]=v;stars[i,j]=(lo>0 or hi<0)
   VALUES.append(dict(figure='increment',scenario=s,epsilon=e,method='mecc_vs_calibrate_public',value=v,low=lo,high=hi,ci_excludes_zero=int(stars[i,j])))
 cmap=colors.LinearSegmentedColormap.from_list('gain',['#be705f','#fafafa','#267b82'])
 norm=colors.TwoSlopeNorm(vmin=-11,vcenter=0,vmax=11)
 im=ax.imshow(data,cmap=cmap,norm=norm,aspect='auto',interpolation='nearest')
 ax.set_yticks(range(6),LABELS);ax.set_xticks(range(6),['0.5','1','2','3','4','5']);ax.set_xlabel(r'$\epsilon$',labelpad=1);ax.tick_params(length=0,pad=3)
 for sp in ax.spines.values():sp.set_visible(False)
 for i in range(6):
  for j in range(6):
   label=f'{data[i,j]:.2f}'+('*' if stars[i,j] else '')
   ax.text(j,i,label,ha='center',va='center',fontsize=9,color='white' if abs(data[i,j])>7 else '#162d37')
 ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,6,1),minor=True);ax.grid(which='minor',color='white',lw=.8);ax.tick_params(which='minor',length=0)
 cax=fig.add_axes([.275,.13,.70,.05]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-10,0,10]);cb.outline.set_visible(False);cb.ax.tick_params(length=2,pad=1);cb.set_label('MSE reduction (%)',labelpad=1)
 save(fig,'increment','Complete 6 x 6 gain map: 100*(Calibrate-public MSE - MECC MSE)/Calibrate-public MSE. Asterisk: unadjusted paired 95% difference interval excludes zero. Symmetric color scale preserves negatives; signed 0.00 denotes rounded small nonzero values, full precision in plot-values.csv.')
def risk():
 fig,(ax,bx)=plt.subplots(1,2,figsize=(WIDTH,2.60),sharey=True,gridspec_kw={'width_ratios':[1,1]});fig.subplots_adjust(left=.165,right=.985,bottom=.20,top=.82,wspace=.18)
 cases=[(s,e) for s in SCENARIOS[:4] for e in [3.,5.]];yy=np.arange(8)[::-1];labs=[]
 for idx,(s,e) in enumerate(cases):
  y=yy[idx];r=RISK[s,e];scale=100/float(SUMMARY[s,e,'calibrate_pl0_public_pre']['mse_mean']);labs.append(f'{LABELS[SCENARIOS.index(s)]}  {e:g}')
  vals={m:scale*float(r[m+'_mean']) for m in ['oracle_gain','coefficient_regret','predicted_pre_mse_delta','actual_pre_mse_delta']}
  ax.barh(y+.16,vals['oracle_gain'],height=.29,color=BLUE,label='Oracle gain' if idx==0 else None,zorder=3)
  ax.barh(y-.16,vals['coefficient_regret'],height=.29,color=GOLD,label='Coefficient regret' if idx==0 else None,zorder=3)
  bx.plot([vals['predicted_pre_mse_delta'],vals['actual_pre_mse_delta']],[y,y],color='#8c959a',lw=1,zorder=3)
  for m,c,mk in [('predicted_pre_mse_delta',BLUE,'o'),('actual_pre_mse_delta',RED,'s')]:
   v=vals[m];lo=scale*float(r[m+'_ci95_low']);hi=scale*float(r[m+'_ci95_high'])
   bx.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt=mk,color=c,ms=3.5,lw=.75,capsize=1.5,zorder=4,label=('Predicted' if m.startswith('predicted') else 'Observed') if idx==0 else None)
  for m,v in vals.items():VALUES.append(dict(figure='risk-evidence',scenario=s,epsilon=e,method=m,value=v,low=scale*float(r[m+'_ci95_low']),high=scale*float(r[m+'_ci95_high'])))
 for a in [ax,bx]:
  a.spines[['top','right','left']].set_visible(False);a.tick_params(axis='y',length=0,pad=5);a.tick_params(axis='x',length=3,pad=2);a.grid(axis='x',color='#e1e5e8',lw=.5,zorder=0);a.set_ylim(-.6,7.6)
  for y in [1.5,5.5]:a.axhspan(y-2,y,facecolor='#f0f4f6',zorder=0)
 ax.set_yticks(yy,labs);ax.set_xlim(0,11);ax.set_xticks([0,5,10]);ax.set_xlabel('Component / base MSE (%)',labelpad=2)
 bx.axvline(0,color='#757e85',lw=.7);bx.set_xlim(-11,1.0);bx.set_xticks([-10,-5,0]);bx.set_xlabel('Risk change / base MSE (%)',labelpad=2)
 fig.text(.025,.85,r'Input       $\epsilon$',fontsize=9)
 ax.text(.01,1.045,'(a)',transform=ax.transAxes);bx.text(.01,1.045,'(b)',transform=bx.transAxes)
 ax.legend(loc='lower left',bbox_to_anchor=(-.02,1.09),ncol=2,frameon=False,columnspacing=.9,handlelength=1.2,handletextpad=.4,borderaxespad=0)
 bx.legend(loc='lower left',bbox_to_anchor=(-.02,1.09),ncol=2,frameon=False,columnspacing=.9,handlelength=1.2,handletextpad=.4,borderaxespad=0)
 save(fig,'risk-evidence','Four primary scenarios at epsilon 3 and 5. Left: oracle gain and coefficient regret. Right: predicted and observed preprojection risk change, connected per condition, with unadjusted 95% t intervals. All normalized by corresponding mean preprojection Calibrate-public MSE. Full epsilon coverage remains in mechanism-summary.csv.')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 for s in SCENARIOS:
  for e in EPS:
   assert int(PAIRED[s,e]['repetitions'])==100
   assert int(RISK[s,e]['repetitions'])==100
   for m in ['mecc','calibrate_pl0_public','calibrate_pl0_public_pre','calibrate_pl0_mle','oue_full_mle']:assert int(SUMMARY[s,e,m]['repetitions'])==100
 mse();heatmap();risk()
 runpy.run_path(str(ROOT/'scripts/plot_mechanism.py'),run_name='__main__')
 runpy.run_path(str(ROOT/'scripts/plot_evidence.py'),run_name='__main__')
 mechanism_manifest=json.loads((OUT/'mechanism-manifest.json').read_text())
 evidence_manifest=json.loads((OUT/'evidence-manifest.json').read_text())
 MANIFEST['figures']['mechanism']={'width_inches':mechanism_manifest['dimensions_inches'][0],'height_inches':mechanism_manifest['dimensions_inches'][1],'files':mechanism_manifest['files'],'description':mechanism_manifest['kind'],'detail_manifest':'mechanism-manifest.json'}
 MANIFEST['figures'].update(evidence_manifest['figures'])
 MANIFEST['figures']={name:MANIFEST['figures'][name] for name in ['mechanism','mse','increment','risk-evidence','coordinate-gains','row-alignment']}
 MANIFEST['additional_manifests']=['mechanism-manifest.json','evidence-manifest.json']
 MANIFEST['evidence_sources']={name:sha(ROOT/'results'/name) for name in ['coordinate-gains.csv','coordinate-gains-per-seed.csv','row-alignment.csv','row-alignment-statistics.csv']}
 with (OUT/'plot-values.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['figure','scenario','epsilon','method','value','low','high','ci_excludes_zero']);w.writeheader();w.writerows(VALUES)
 MANIFEST['plot_values_sha256']=sha(OUT/'plot-values.csv')
 (OUT/'manifest.json').write_text(json.dumps(MANIFEST,indent=2)+'\n')
 print(json.dumps({'figures':list(MANIFEST['figures']),'plot_values':len(VALUES),'font':FAMILY}))
if __name__=='__main__':main()
