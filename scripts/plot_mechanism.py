"""Exact constructed illustrations of MECC; no empirical observations are used."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import binom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / 'outputs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT = 516 / 72.27, 4.50
BLUE, GREEN, RED, GOLD = '#27698d', '#27766b', '#b44332', '#b48a36'
INK, LIGHT, LINE = '#243943', '#f1f5f7', '#c6d0d6'
try:
    FONT = font_manager.findfont('Times New Roman', fallback_to_default=False)
    FAMILY = 'Times New Roman'
except ValueError:
    FONT = font_manager.findfont('DejaVu Serif')
    FAMILY = 'DejaVu Serif'
plt.rcParams.update({'font.family':FAMILY,'font.size':9.5,
                    'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,
                    'legend.fontsize':9,'mathtext.fontset':'stix',
                    'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.65,
                    'xtick.major.width':.65,'ytick.major.width':.65,
                    'savefig.dpi':300})

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def text(ax,x,y,s,**kw):
    kw.setdefault('color',INK);kw.setdefault('fontsize',9.5)
    return ax.text(x,y,s,**kw)

def arrow(ax,a,b,color=INK,**kw):
    obj=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,
                       lw=.85,color=color,**kw)
    ax.add_patch(obj)
    return obj

def matrix(ax,M,x,y,w,h,cmap='binary',labels=False,highlight=False):
    """x,y denote lower-left; rows are displayed in their given order."""
    M=np.asarray(M);nr,nc=M.shape
    for i in range(nr):
        for j in range(nc):
            xx=x+j*w/nc;yy=y+(nr-1-i)*h/nr
            if cmap=='training': color=('#c3dce9' if i<3 else '#cce4dd') if M[i,j] else '#fff'
            else: color='#c3dce9' if M[i,j] else '#fff'
            ax.add_patch(Rectangle((xx,yy),w/nc,h/nr,facecolor=color,edgecolor='#7b929e',lw=.55))
            if labels: text(ax,xx+w/(2*nc),yy+h/(2*nr),str(M[i,j]),ha='center',va='center')
    if highlight:
        ax.add_patch(Rectangle((x,y+(nr-1)*h/nr),w,h/nr,fill=False,ec=RED,lw=1.25))

def public_base(n,d,epsilon):
    k=np.arange(n+1,dtype=float); logk=np.log1p(k)
    def prior(alpha):
        z=-alpha*logk
        return np.exp(z-logsumexp(z))
    hi=2.
    while prior(hi)@k>n/d: hi*=2
    alpha=brentq(lambda a:prior(a)@k-n/d,0.,hi,xtol=1e-12) if d>2 else 0.
    weights=prior(alpha);q=1/(np.exp(epsilon)+1);delta=.5-q;v=q*(1-q)/delta**2
    counts=np.arange(n+1,dtype=float)
    u=(counts-n*q)/delta
    ll=-.5*(u[:,None]-k[None,:])**2/(n*v)
    ll-=np.max(ll,axis=1)[:,None];lik=np.exp(ll)
    g=(lik@(weights*k))/(lik@weights)/n
    return g,alpha,q,delta,v

def project_simplex(x):
    u=np.sort(np.asarray(x))[::-1]
    css=np.cumsum(u)-1
    rho=np.flatnonzero(u-css/np.arange(1,len(u)+1)>0)[-1]
    return np.maximum(np.asarray(x)-css[rho]/(rho+1),0)

def generate():
    n,d,epsilon,n_A,m,C_Aj,k_j=64,8,3.,16,48,5,10
    g,alpha,q,delta,v=public_base(n,d,epsilon)
    S=np.arange(m)
    kernel=np.convolve(binom.pmf(np.arange(k_j+1),k_j,.5),
                       binom.pmf(np.arange(m-k_j),m-1-k_j,q))
    dg=np.diff(g)[C_Aj+S]
    mu=float(kernel@dg)
    M1=np.array([[1,1],[1,0],[0,1],[0,0]])
    M2=np.array([[1,0],[1,0],[0,1],[0,1]])
    controls=np.array([12,8,6],dtype=float)/m
    D=v*(v+2*(k_j/m+controls))/m
    c=delta*v*controls*mu
    beta_star=c/D
    example_b=1.6
    assert np.all(M1.sum(0)==M2.sum(0))
    assert np.prod(M1,axis=1).sum()-np.prod(M2,axis=1).sum()==1
    assert abs(kernel.sum()-1)<1e-12 and len(kernel)==m
    assert np.all(D>0) and np.all(np.diff(g)>=-1e-14)
    assert abs((example_b**2-2*example_b)-((example_b-1)**2-1))<1e-14
    # Direct finite sums independently check the displayed covariance identity.
    direct_cov=[]
    for count_k in (12,8,6):
        total=0.
        for kind,number in [('j',k_j),('k',count_k),('other',m-k_j-count_k)]:
            leave_j=k_j-int(kind=='j')
            ps=np.convolve(binom.pmf(np.arange(leave_j+1),leave_j,.5),
                           binom.pmf(np.arange(m-leave_j),m-1-leave_j,q))
            pj=.5 if kind=='j' else q
            pk=.5 if kind=='k' else q
            term=0.
            for yj in (0,1):
                for yk in (0,1):
                    probability=(pj if yj else 1-pj)*(pk if yk else 1-pk)
                    term+=probability*(yj-q)*(yk-q)/delta**2*float(ps@g[C_Aj+S+yj])
            total+=number/m*term
        direct_cov.append(total)
    assert np.allclose(direct_cov,c,rtol=1e-11,atol=1e-14)

    fig=plt.figure(figsize=(WIDTH,HEIGHT))
    panels=[fig.add_axes(r) for r in [(.022,.542,.455,.437),(.522,.542,.455,.437),
                                    (.022,.025,.455,.455),(.522,.025,.455,.455)]]
    for ax in panels:
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    # Thin separators express four complementary mathematical views.
    fig.add_artist(plt.Line2D([.5,.5],[.025,.975],transform=fig.transFigure,color=LINE,lw=.7))
    fig.add_artist(plt.Line2D([.025,.975],[.51,.51],transform=fig.transFigure,color=LINE,lw=.7))

    # (a) A known zero moment and an exact same-marginal counterexample.
    ax=panels[0]
    text(ax,0,.99,'(a)',weight='bold',va='top')
    text(ax,.09,.99,'Known moments beyond column sums',va='top')
    text(ax,.03,.83,r'$\mathbb{E}[Z_{ij}Z_{ik}]=\mathbf{1}\{x_i=j\}\mathbf{1}\{x_i=k\}=0$')
    text(ax,.03,.72,r'$Z_{ij}=(Y_{ij}-q)/\delta,\quad j\ne k$')
    for M,x in [(M1,.12),(M2,.64)]:
        matrix(ax,M,x,.29,.18,.32,labels=True,highlight=M is M1)
        text(ax,x+.045,.65,'$j$',ha='center')
        text(ax,x+.135,.65,'$k$',ha='center')
        text(ax,x+.09,.21,r'$C_B=(2,2)$',ha='center')
    arrow(ax,(.35,.45),(.59,.45),color=RED)
    text(ax,.47,.55,'Permute $k$',ha='center')
    text(ax,.21,.12,'1 intersection',color=RED,ha='center')
    text(ax,.73,.12,'0 intersections',color=RED,ha='center')
    ax.add_patch(Rectangle((.02,.005),.96,.075,fc=LIGHT,ec='none'))
    text(ax,.5,.042,r'$G^{(1)}=G^{(2)},\qquad T_{jk}^{(1)}-T_{jk}^{(2)}=1/(4\delta^2)$',ha='center',va='center')

    # (b) Information flow, with actual matrix and coefficient objects.
    ax=panels[1]
    text(ax,0,.99,'(b)',weight='bold',va='top')
    text(ax,.09,.99,'Learn on $A$; retain all reports in $G$',va='top')
    schematic=np.array([[1,1,0,1],[0,1,0,0],[1,0,1,0],
                        [1,1,0,0],[1,0,0,1],[0,1,1,0],[0,0,1,0],[1,0,1,1]])
    matrix(ax,schematic,.065,.31,.16,.49,cmap='training')
    ax.plot([.05,.24],[.617,.617],color=INK,lw=.75)
    text(ax,.025,.73,'$A$',color=BLUE,ha='center')
    text(ax,.025,.45,'$B$',color=GREEN,ha='center')
    text(ax,.145,.84,'OUE reports',ha='center')
    # Learned frequency object and selected controls.
    arrow(ax,(.235,.73),(.31,.73),BLUE)
    g_A=public_base(3,4,epsilon)[0]
    rs=project_simplex(g_A[schematic[:3].sum(0)])
    selected={0,1}
    for j,h in enumerate(rs):
        ax.add_patch(Rectangle((.33+j*.04,.66),.029,.31*h,fc=BLUE if j in selected else '#c0d3dc',ec='none'))
    text(ax,.41,.82,'$r$',ha='center',color=BLUE)
    text(ax,.41,.59,r'$\Pi_\Delta g_{n_A}(C_A)$',ha='center')
    arrow(ax,(.50,.73),(.56,.73),BLUE)
    text(ax,.64,.78,'$H$',ha='center',color=BLUE)
    text(ax,.64,.665,r'$\widetilde\mu(C_A,r)$',ha='center',color=BLUE,fontsize=9)
    arrow(ax,(.72,.73),(.78,.73),BLUE)
    text(ax,.88,.74,r'$\widehat\beta$',ha='center',color=BLUE)
    text(ax,.88,.63,r'fixed given $A$',ha='center',color=BLUE,fontsize=9)
    arrow(ax,(.88,.60),(.88,.48),BLUE)
    # B-only moment branch, with bit intersections and the correction.
    arrow(ax,(.235,.43),(.32,.43),GREEN)
    text(ax,.44,.45,r'$C_{B,jk}$',ha='center',color=GREEN)
    text(ax,.44,.35,'aligned AND',ha='center',color=GREEN,fontsize=9)
    arrow(ax,(.56,.43),(.61,.43),GREEN)
    text(ax,.68,.43,r'$T_{jk}$',ha='center',color=GREEN)
    arrow(ax,(.75,.43),(.81,.43),GREEN)
    text(ax,.88,.43,r'$\sum_k\widehat\beta_{jk}T_{jk}$',ha='center',fontsize=9)
    # The base estimator receives both sets, independently of the correction.
    arrow(ax,(.145,.30),(.145,.22))
    text(ax,.145,.17,r'$C_A+C_B$',ha='center')
    arrow(ax,(.28,.20),(.33,.20))
    text(ax,.40,.18,r'$g_n$',ha='center')
    arrow(ax,(.45,.20),(.50,.20))
    text(ax,.55,.18,r'$G$',ha='center')
    arrow(ax,(.60,.20),(.67,.20))
    text(ax,.72,.18,r'$W$',ha='center',color=RED)
    arrow(ax,(.78,.20),(.86,.20),RED)
    text(ax,.93,.18,r'$\widehat f$',ha='center',color=RED)
    text(ax,.82,.255,r'$\Pi_\Delta$',ha='center',color=RED,fontsize=9)
    ax.plot([.88,.72],[.36,.36],color=RED,lw=.85)
    arrow(ax,(.72,.36),(.72,.25),RED)
    text(ax,.79,.375,r'$-$',color=RED,ha='center')
    ax.add_patch(Rectangle((.02,.005),.96,.075,fc=LIGHT,ec='none'))
    text(ax,.5,.042,r'$\mathbb{E}_A T_{jk}=0\quad\Longrightarrow\quad\mathbb{E}_A W_j=\mathbb{E}_A G_j$',ha='center',va='center')

    # (c) A public nonlinear table and the actual leave-one-out channel.
    ax=panels[2]
    text(ax,0,.99,'(c)',weight='bold',va='top')
    text(ax,.09,.99,'Discrete covariance from finite differences',va='top')
    gx=ax.inset_axes([.115,.40,.34,.36])
    gx.plot(np.arange(n+1),g,color=BLUE,lw=1.3)
    gx.set(xlim=(0,32),ylim=(0,1),xticks=[0,16,32],yticks=[0,.5,1])
    gx.set_xlabel('$c$',labelpad=0);gx.set_ylabel('$g_n(c)$',labelpad=0)
    gx.tick_params(pad=1,length=2)
    gx.spines[['top','right']].set_visible(False)
    marked=12
    gx.plot([marked,marked+1],g[[marked,marked+1]],'o',color=GOLD,ms=3)
    gx.annotate(r'$\Delta g_n$',xy=(marked+.5,float(g[marked:marked+2].mean())),xytext=(19,.30),
                fontsize=9,color=GOLD,arrowprops={'arrowstyle':'-','color':GOLD,'lw':.7})
    px=ax.inset_axes([.62,.40,.34,.36])
    px.bar(S,kernel,color=GREEN,width=.8,linewidth=0)
    px.set(xlim=(-.5,20.5),ylim=(0,.21),xticks=[0,10,20],yticks=[0,.1,.2])
    px.set_xlabel('$s$',labelpad=0);px.set_ylabel(r'$\Pr(S_j=s)$',labelpad=0)
    px.tick_params(pad=1,length=2);px.spines[['top','right']].set_visible(False)
    text(ax,.5,.85,r'$S_j\sim\mathrm{Bin}(10,p)+\mathrm{Bin}(37,q)$',ha='center')
    text(ax,.5,.22,r'$\mu_j=\sum_s\Pr(S_j=s)\,[g_n(C_{Aj}+s+1)-g_n(C_{Aj}+s)]$',ha='center',fontsize=9)
    text(ax,.5,.115,r'$c_{jk}=\delta v f_{Bk}\mu_j$',ha='center',color=BLUE)
    text(ax,.5,.025,r'$n=64,\ d=8,\ \varepsilon=3,\ m=48,\ C_{Aj}=5$',ha='center',fontsize=9)

    # (d) Orthogonality and the exact per-direction risk quadratic.
    ax=panels[3]
    text(ax,0,.99,'(d)',weight='bold',va='top')
    text(ax,.09,.99,'Orthogonal controls; separate scalar choices',va='top')
    text(ax,.19,.82,r'$\mathrm{Cov}_A(T_j)$',ha='center')
    x,y,w=.055,.48,.085
    for i in range(3):
        for j in range(3):
            xx=x+j*w;yy=y+(2-i)*w
            ax.add_patch(Rectangle((xx,yy),w,w,fc='#c3dce9' if i==j else '#f5f7f8',ec=LINE,lw=.5))
            text(ax,xx+w/2,yy+w/2,f'$D_{i+1}$' if i==j else '$0$',ha='center',va='center')
    text(ax,.19,.385,r'$\beta^*_{jk}=c_{jk}/D_{jk}$',ha='center')
    text(ax,.19,.285,'one target $j$',ha='center',fontsize=9)
    rx=ax.inset_axes([.46,.40,.51,.38])
    t=np.linspace(0,2.15,240);risk=t*t-2*t
    rx.plot(t,risk,color=BLUE,lw=1.3)
    rx.fill_between(t,risk,0,where=risk<=0,color='#e7f1f4')
    rx.axhline(0,color='#8a969d',lw=.65)
    rx.plot([1],[-1],'o',color=BLUE,ms=4)
    example_risk=example_b**2-2*example_b
    rx.plot([example_b],[example_risk],'s',color=RED,ms=4)
    rx.plot([1,example_b],[-1,-1],ls=':',color=GOLD,lw=.85)
    rx.annotate('',xy=(example_b,example_risk),xytext=(example_b,-1),
                arrowprops={'arrowstyle':'<->','color':GOLD,'lw':.8,'shrinkA':0,'shrinkB':0})
    rx.text(1.90,-.88,'regret\n/ gain',ha='center',va='center',fontsize=9,color=GOLD,linespacing=1.0)
    rx.text(.85,-1.13,'oracle',fontsize=9,color=BLUE,ha='right')
    rx.text(1.62,-.40,'example',fontsize=9,color=RED,ha='right')
    rx.set(xlim=(0,2.15),ylim=(-1.22,.48),xticks=[0,1,2],yticks=[-1,0])
    rx.tick_params(pad=1,length=2);rx.spines[['top','right']].set_visible(False)
    rx.set_xlabel(r'$b=\beta/\beta^*$',labelpad=0)
    rx.text(.03,1.04,r'$\Delta R\,/(c^2/D)$',transform=rx.transAxes,fontsize=9)
    rx.text(.98,1.04,r'$c\ne0$',transform=rx.transAxes,fontsize=9,ha='right')
    text(ax,.5,.17,r'$\Delta R_j=-\sum_k c_{jk}^2/D_{jk}+\sum_kD_{jk}(\beta_{jk}-\beta^*_{jk})^2$',ha='center',fontsize=9)
    text(ax,.5,.055,'oracle gain                     coefficient regret',ha='center',fontsize=9)
    text(ax,.5,.005,'Before projection',ha='center',color='#6a7880',fontsize=9)

    fig.canvas.draw()
    texts=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            bb=t.get_window_extent(fig.canvas.get_renderer())
            texts.append({'text':t.get_text(),'fontsize':t.get_fontsize(),
                          'bounds_px':[bb.x0,bb.y0,bb.x1,bb.y1]})
    for suffix in ['png','pdf']:
        fig.savefig(OUT/f'mechanism.{suffix}',dpi=300,facecolor='white')
    plt.close(fig)
    record={'kind':'Exact constructed mathematical illustrations; not empirical observations',
            'dimensions_inches':[WIDTH,HEIGHT],'dpi':300,'font':{'family':FAMILY,'file_name':Path(FONT).name,'sha256':sha(FONT)},
            'script_sha256':sha(__file__),
            'constructive_matrices':{'matrices':[M1.tolist(),M2.tolist()],'column_sums':[M1.sum(0).tolist(),M2.sum(0).tolist()],
                                     'intersections':[1,0],'control_difference':'1/(4 delta^2)'},
            'dependency_panel':{'report_matrix':schematic.tolist(),'n':8,'d':4,'n_A':3,'epsilon':epsilon,
                                'training_frequency_bars':rs.tolist(),'selected_control_indices':[0,1],
                                'bar_rule':'Actual public Calibrate estimate and simplex projection of shown three-row training matrix; top-2 emphasized.',
                                'figure_role':'Constructed information-flow example; distinct from panel-c illustration.',
                                'base_information':'A and B column sums','coefficient_information':'A and public quantities only',
                                'control_information':'aligned B report rows'},
            'finite_difference_example':{'n':n,'d':d,'epsilon':epsilon,'n_A':n_A,'m':m,'C_Aj':C_Aj,'mf_Bj':k_j,
                                         'q':q,'delta':delta,'v':v,'public_prior_alpha':alpha,
                                         'prior_mean':n/d,'kernel_mass':float(kernel.sum()),'mu':mu,
                                         'S':S.tolist(),'kernel':kernel.tolist(),'g_n':g.tolist(),'finite_differences':dg.tolist(),
                                         'displayed_count_range':[0,32],
                                         'displayed_kernel_range':[0,20],'probability_mass_outside_display':float(kernel[21:].sum()),
                                         'control_frequencies':controls.tolist(),'control_variances':D.tolist(),
                                         'control_covariances':c.tolist(),'direct_finite_sum_covariances':direct_cov,
                                         'oracle_coefficients':beta_star.tolist()},
            'risk_quadratic':{'scope':'One nonzero-covariance control direction, conditional and before projection',
                              'x':'b = beta / beta_star','y':'Delta risk / (c^2/D)',
                              'function':'b^2 - 2 b = (b-1)^2 - 1','illustrative_b':example_b,
                              'illustrative_regret':(example_b-1)**2,'illustrative_risk_delta':example_risk},
            'checks':{'matrix_columns_equal':True,'intersection_difference_one':True,'kernel_mass_one':True,
                      'positive_control_variances':True,'direct_covariance_identity':True,
                      'risk_square_identity':True,'training_bars_match_shown_reports':True,'no_new_experiment':True},
            'visible_text':texts,'files':{f'mechanism.{s}':sha(OUT/f'mechanism.{s}') for s in ['png','pdf']}}
    (OUT/'mechanism-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'figures':['mechanism.pdf','mechanism.png'],'size':[WIDTH,HEIGHT],
                      'mu':mu,'kernel_display_tail':float(kernel[21:].sum())}))

if __name__=='__main__':
    generate()
