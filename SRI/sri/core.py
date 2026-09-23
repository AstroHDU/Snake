from __future__ import annotations
import numpy as np

PDIM = 4  # affine basis: 1 + X + Y + Z


def _tree(points):
    p=np.asarray(points,float); n=len(p)
    if p.ndim!=2 or not np.isfinite(p).all():
        raise ValueError('finite point matrix required')
    d=np.linalg.norm(p[:,None]-p[None,:],axis=-1)
    seen=np.zeros(n,bool)
    if n: seen[0]=True
    edges=[]
    for _ in range(n-1):
        best=(np.inf,-1,-1)
        for i in np.flatnonzero(seen):
            av=np.flatnonzero(~seen)
            j=int(av[np.argmin(d[i,av])])
            if d[i,j] < best[0]: best=(float(d[i,j]),int(i),j)
        L,i,j=best; seen[j]=True; edges.append((i,j,L))
    return sorted(edges,key=lambda e:e[2],reverse=True),d


def cen(points, clouds=None, tested_edge_floor=True, width_model='smem_scatter'):
    """Longest physical edge relative to whole-tree Q75 and cloud extent."""
    p=np.asarray(points,float)
    if len(p)<3:return {'value':np.nan,'status':'insufficient_centers','edges':[]}
    if clouds is not None and len(clouds)!=len(p):raise ValueError('cloud/point count mismatch')
    ed,_=_tree(p);i,j,L=ed[0]
    if L==0:return {'value':1.,'status':'computed','maximum_index':0,'edges':[]}
    b=float(np.quantile([e[2] for e in ed[1:]],.75));wt=wr=0.
    if clouds is not None:
        u=(p[j]-p[i])/L
        if width_model=='smem_scatter':
            from sri_preprocess.fixed_components import scatter
            widths=np.array([np.sqrt(u@scatter(c)@u) for c in clouds])
        else:raise ValueError('Unknown spatial width model')
        wt=float(widths[i]+widths[j]);wr=float(np.median([widths[a]+widths[b_] for a,b_,_ in ed[1:]]))
    B=max(b,wr,wt) if tested_edge_floor else max(b,wr)
    term=dict(i=i,j=j,length=L,reference=b,width_reference=wr,width_tested=wt,reference_used=B,nref=len(ed)-1)
    return {'value':float(np.sqrt(min(1.,B/L))),'status':'computed','maximum_index':0,'edges':[term]}


def vcen_local(points, covariances):
    """Longest physical UVW edge, whole-tree Q75, finite excess-error relief."""
    v=np.asarray(points,float);C=np.asarray(covariances,float);n=len(v)
    if n<3:return {'value':np.nan,'status':'insufficient_centers','edges':[]}
    if C.shape!=(n,3,3) or not np.isfinite(C).all():raise ValueError('Vcen covariance unavailable: select nominal mode explicitly upstream')
    edges,_=_tree(v);lengths=np.array([e[2] for e in edges]);L=float(lengths[0])
    if L==0:return {'value':1.,'status':'computed','edges':[]}
    grads=np.zeros((len(edges),n,3))
    for k,(i,j,l) in enumerate(edges):
        if l>0: u=(v[j]-v[i])/l;grads[k,i]=-u;grads[k,j]=u
    # Zero-length references cannot identify a positive relative velocity scale.
    ids=np.flatnonzero(lengths[1:]>0)+1
    if len(ids)==0:raise ValueError('Vcen has no positive reference edge')
    order=ids[np.argsort(lengths[ids])];h=.75*(len(order)-1);lo=int(np.floor(h));hi=int(np.ceil(h));f=h-lo
    low=lengths[order[lo]];high=lengths[order[hi]];b=float((1-f)*low+f*high)
    g=grads[0]-((1-f)*grads[ids[lengths[ids]==low]].mean(0)+f*grads[ids[lengths[ids]==high]].mean(0))
    var=max(0.,float(np.einsum('ni,nij,nj->',g,C,g)));d=max(0.,L-b)/b;q=var/b**2
    floor=min(d*d,1.);z=float(np.sqrt(floor+(d*d-floor)/(1+q)))
    i,j,_=edges[0];term=dict(i=i,j=j,length=L,reference=b,difference_sigma=float(np.sqrt(var)),nref=len(ids),excess=z)
    return {'value':float(1/np.sqrt(1+z)),'status':'computed','maximum_index':0,'edges':[term]}


def affine_weights(x, ridge=0.1):
    """Native scalar affine LOO operator used only for geometry amplification."""
    x=np.asarray(x,float); n=len(x)
    z=x-x.mean(0); s=np.sqrt(np.mean(np.sum(z*z,axis=1))); z=z/(s if s>0 else 1.)
    W=np.zeros((n,n)); identifiable=[]; design=np.c_[np.ones(n),z]
    for i in range(n):
        tr=np.arange(n)!=i; nt=int(tr.sum()); zt=z[tr]; zc=zt-zt.mean(0)
        W[i,tr]=1/nt+(z[i]-zt.mean(0))@np.linalg.solve(zc.T@zc+ridge*nt*np.eye(3),zc.T)
        pr=np.linalg.pinv(design[tr],rcond=1e-10)@design[tr]
        identifiable.append(bool(np.allclose(design[i]@pr,design[i],rtol=1e-8,atol=1e-9)))
    return W,np.asarray(identifiable,bool)


def _design(x):
    x=np.asarray(x,float)
    z=x-x.mean(0); s=np.sqrt(np.mean(np.sum(z*z,axis=1))); z=z/(s if s>0 else 1.)
    return np.array([np.kron(np.r_[1.,q][None,:],np.eye(3)) for q in z])


def _train_fit(A,v,tr,ridge,fixed=None):
    """Geometry-only block ridge fit with optional fixed training weights."""
    n=len(v); mask=np.zeros(n,bool); mask[np.asarray(tr,int)]=True
    I=np.eye(3); P=np.tile(I[None,:,:],(n,1,1))
    N=np.einsum('nai,nab,nbj->nij',A,P,A)
    rhs=np.einsum('nai,nab->nib',A,P)
    b=np.einsum('nib,nb->ni',rhs,v)
    pen=np.kron(np.diag([0.,1.,1.,1.]),ridge*P[mask].sum(0))
    w=mask.astype(float) if fixed is None else mask*np.asarray(fixed,float)
    H=np.einsum('n,nij->ij',w,N)+pen
    beta=np.linalg.solve(H,np.einsum('n,ni->i',w,b))
    # Frozen-weight linear operator for conditional measurement propagation.
    K=np.linalg.solve(H,(w[:,None,None]*rhs).transpose(1,0,2).reshape(12,3*n))
    predop=np.einsum('nai,ij->naj',A,K).reshape(n,3,n,3).transpose(0,2,1,3)
    pred=np.einsum('ijab,jb->ia',predop,v)
    return dict(pred=pred,W=predop,w=w,beta=beta)


def _neff(w):
    w=np.asarray(w,float); s=float(w.sum()); q=float(w@w)
    return s*s/q if q>0 else 0.0


def _support_blend(w,target):
    """Blend fixed training weights toward one until Kish Neff >= target."""
    w=np.asarray(w,float); n=len(w)
    if target<=0 or _neff(w)>=target: return w,1.0
    if target>=n-1e-12: return np.ones(n),0.0
    lo,hi=0.0,1.0
    for _ in range(70):
        alpha=(lo+hi)/2; z=1-alpha*(1-w)
        if _neff(z)>=target: lo=alpha
        else: hi=alpha
    return 1-lo*(1-w),lo


def _pred_loss_geometry(A,v,training,target,ridge):
    f=_train_fit(A,v,training,ridge)
    r=v[target]-f['pred'][target]
    return float(r@r)


def _exact_adaptive_weights(x,v,outer,ridge):
    """Training-only high-M influence localization; outer target is never used."""
    n=len(v); A=_design(x); tr=np.array([j for j in range(n) if j!=outer],int); nt=len(tr)
    if nt-2<PDIM: return None
    base={k:_pred_loss_geometry(A,v,[q for q in tr if q!=k],k,ridge) for k in tr}
    influence=[]
    for j in tr:
        total=0.0
        for k in tr:
            if k==j: continue
            after=_pred_loss_geometry(A,v,[q for q in tr if q!=j and q!=k],k,ridge)
            total+=max(0.0,base[k]-after)
        influence.append(total)
    influence=np.asarray(influence,float); total=float(influence.sum())
    p=influence/total if total>1e-15 else np.full(nt,1/nt)
    excess=np.maximum(p-1/nt,0.0); mx=float(excess.max())
    w=np.ones(nt) if mx<=1e-15 else 1-excess/mx
    w,alpha=_support_blend(w,PDIM)
    return tr,w,dict(method='exact',p=p,influence=influence,pmax=float(p.max()),Keff=float(1/max(p@p,1e-30)),neff=float(_neff(w)),minw=float(w.min()),alpha=float(alpha))


def _scalar_design(x):
    x=np.asarray(x,float); z=x-x.mean(0); s=np.sqrt(np.mean(np.sum(z*z,axis=1))); z=z/(s if s>0 else 1.0)
    return np.c_[np.ones(len(x)),z]


def _analytic_lowm_weights(x,v,outer,ridge):
    """Low-M ridge/PRESS-Cook influence; no nested deletion is required."""
    X=_scalar_design(x); tr=np.array([j for j in range(len(v)) if j!=outer],int)
    Xt=X[tr]; Y=v[tr]; nt=len(tr); D=np.diag([0.,1.,1.,1.])
    inv=np.linalg.inv(Xt.T@Xt+ridge*nt*D); Hm=Xt@inv@Xt.T; fit=Hm@Y; residual=Y-fit
    hdiag=np.clip(np.diag(Hm),0,1-1e-12); influence=[]
    for a in range(nt):
        den=max(1-hdiag[a],1e-12); e=residual[a]; hc=Hm[:,a].copy(); hc[a]=0.0
        influence.append((e@e)/(den*den)*(hc@hc))
    influence=np.asarray(influence,float); mean=float(influence.mean()) if influence.mean()>1e-15 else 1.0
    w=1/np.sqrt(1+influence/mean); df=float(np.trace(Hm)); w,alpha=_support_blend(w,df)
    total=float(influence.sum()); p=influence/total if total>1e-15 else np.full(nt,1/nt)
    return tr,w,dict(method='analytic',p=p,influence=influence,pmax=float(p.max()),Keff=float(1/max(p@p,1e-30)),neff=float(_neff(w)),minw=float(w.min()),alpha=float(alpha),effective_df=df)


def _influence_weights(x,v,outer,ridge):
    exact=_exact_adaptive_weights(x,v,outer,ridge)
    return exact if exact is not None else _analytic_lowm_weights(x,v,outer,ridge)


def _tangent_radial_parts(z,E,nlos,h):
    """Split normalized LOO residual into local tangential/radial evidence."""
    nlos=np.asarray(nlos,float); norm=np.linalg.norm(nlos)
    if norm<=0 or not np.isfinite(norm): raise ValueError('finite nonzero line-of-sight direction required')
    nlos=nlos/norm; P=np.eye(3)-np.outer(nlos,nlos)
    # Diagonalize within the actual tangent plane. Selecting the largest two
    # eigenvectors of a 3D zero/rank-deficient matrix can include the radial axis.
    _,_,basis=np.linalg.svd(nlos.reshape(1,3),full_matrices=True)
    T=basis[1:].T
    Et=T.T@E@T; Et=.5*(Et+Et.T)
    eig,U=np.linalg.eigh(Et)
    eig_t=np.maximum(eig,0.0); Ut=T@U; zt=Ut.T@z; r2_t=zt*zt
    zr=float(nlos@z); r2_r=zr*zr; sig2_r=max(float(nlos@E@nlos),0.0)
    hd2=h*h/3.0; cred_t=hd2/(hd2+eig_t)
    loss_t=float((np.minimum(r2_t,hd2)+cred_t*np.maximum(r2_t-hd2,0.0)).sum())
    return dict(tan2=float(r2_t.sum()),rad2=float(r2_r),sig2_r=sig2_r,loss_t=loss_t,eig_t=eig_t,cred_t=cred_t,zr=zr)


def _sqrt_resolved_measurement_loss(z,E,nlos,M,h):
    """Directional uncertainty-aware residual evidence used by SRI.

    Tangential and radial evidence are separated in the entity sightline frame.
    Measurement uncertainty can weaken only residual evidence above the existing
    per-direction working scale h^2/3.  The radial uncertainty relief is then
    moderated continuously by tangential compatibility and by the redundancy
    beyond the four affine basis degrees of freedom.  No new fitted scale is
    introduced.
    """
    p=_tangent_radial_parts(z,E,nlos,h); hd2=h*h/3.0; ht2=2*hd2
    r2=p['rad2']; sig2=p['sig2_r']; excess=max(r2-hd2,0.0)
    cred0=hd2/(hd2+sig2)
    frac=r2/(r2+sig2) if r2+sig2>0 else 1.0
    eta=np.sqrt(frac); cred_r=cred0*eta
    radial_corrected=min(r2,hd2)+cred_r*excess
    q_t=p['tan2']/ht2
    tangent_gate0=1/np.sqrt(1+q_t)
    redundancy=max(0.0,(M-PDIM)/M)
    relief_gate=1-redundancy*(1-tangent_gate0)
    radial_loss=r2-relief_gate*(r2-radial_corrected)
    loss=p['loss_t']+radial_loss
    return float(loss),{**p,'hd2':hd2,'ht2':ht2,'cred0_r':cred0,'resolved_fraction':frac,'eta_r':float(eta),'cred_r':float(cred_r),'q_t':float(q_t),'tangent_gate0':float(tangent_gate0),'redundancy':float(redundancy),'relief_gate':float(relief_gate),'radial_corrected':float(radial_corrected),'radial_loss':float(radial_loss)}


def cross_loss_measurement(x,v,cov,h=5.0,ridge=0.1,cap_single_largest=False,aggregation='mean_residual'):
    """Uncertainty-aware global Cross relation.

    The LOO fit uses training-only influence localization (exact for M>=7 and a
    ridge/PRESS-Cook analytic form at lower multiplicity).  The native affine
    geometry amplification ``a_i`` remains fixed.  Analytic entity-velocity
    covariance enters only after the geometric fit, in the final residual
    evidence layer. The released score averages adjusted residual magnitudes.
    No largest-residual capping or alternate aggregation is used.
    """
    x=np.asarray(x,float); v=np.asarray(v,float); C=np.asarray(cov,float); n=len(v)
    if n<4:
        return {'D_cross_kms':np.nan,'status':'insufficient_centers','loss':np.empty(0),'adjusted_loss':np.empty(0),'diagnostics':[]}
    if x.shape!=v.shape or C.shape!=(n,3,3) or not np.isfinite(x).all() or not np.isfinite(v).all():
        raise ValueError('invalid Cross input shapes/values')
    # Nominal mode supplies explicit zeros upstream; missing data never do.
    if not np.isfinite(C).all():
        raise ValueError('Cross covariance unavailable: select nominal mode explicitly upstream')
    C=.5*(C+C.transpose(0,2,1))
    A=_design(x); W0,ident=affine_weights(x,ridge); Q0=np.eye(n)-W0; a0=np.sum(Q0*Q0,axis=1)
    losses=[]; rows=[]; raw_res=[]
    for i in range(n):
        tr,w,diag=_influence_weights(x,v,i,ridge); fixed=np.zeros(n); fixed[tr]=w
        f=_train_fit(A,v,tr,ridge,fixed=fixed)
        q=-f['W'][i].copy(); q[i]+=np.eye(3)
        raw=np.einsum('jab,jb->a',q,v); z=raw/np.sqrt(a0[i])
        E=np.einsum('jab,jbc,jdc->ad',q,C,q)/a0[i]; E=.5*(E+E.T)
        nlos=x[i]/np.linalg.norm(x[i])
        loss,md=_sqrt_resolved_measurement_loss(z,E,nlos,n,h)
        losses.append(loss); raw_res.append(raw)
        rows.append({**diag,**md,'target':i,'raw_residual':raw,'normalized_residual':z,'residual_covariance':E,'geometry_amplification':float(a0[i]),'loss':float(loss)})
    loss=np.asarray(losses,float); adjusted=loss.copy()
    if aggregation!='mean_residual' or cap_single_largest:raise ValueError('Only uncapped mean residual Cross is supported')
    D=float(np.mean(np.sqrt(loss)))
    return {
        'D_cross_kms':D,'status':'computed','loss':loss,'adjusted_loss':adjusted,
        'diagnostics':rows,'amplification':a0,'prediction_identifiable':ident,
        'raw_rms_kms':float(np.sqrt(np.mean(np.sum(np.asarray(raw_res)**2,axis=1)))),
    }


def harmonic(a,b):
    a=float(a); b=float(b)
    if not (np.isfinite(a) and np.isfinite(b) and 0<=a<=1 and 0<=b<=1):
        raise ValueError('invalid harmonic inputs')
    if a==0 or b==0: return 0.
    return float(2*a*b/(a+b))


def combine(sm,sc,vm,vc,D_cross_kms,M,h_cross_kms):
    S=float(sm if M<3 else np.sqrt(sm*sc))
    V=float(vm if M<3 else np.sqrt(vm*vc))
    R0=harmonic(S,V)
    group='A' if M<3 else 'B' if M==3 else 'C'
    if M<4:
        return {'S':S,'V':V,'R0':R0,'Cross':np.nan,'SRI':R0,'group':group}
    if not np.isfinite(D_cross_kms) or D_cross_kms<0:
        raise ValueError('active Cross requires nonnegative finite D_cross')
    C=float(1/np.hypot(1,D_cross_kms/h_cross_kms))
    sri=float(R0*C)
    return {'S':S,'V':V,'R0':R0,'Cross':C,'SRI':sri,'group':group}
