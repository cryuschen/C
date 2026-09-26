"""Blocked nested selection, sufficient-statistic screening, robust refitting."""
import numpy as np
from scipy.linalg import solve
from .model import design_bank, penalties

LAMBDAS=np.array([.001,.01,.1,1.,10.])


class Bank:
    def __init__(self,ds,params,context=False,design_function=None):
        self.ds=ds;self.params=params;self.context=context;self.dim=18 if context else 9
        self.design=design_bank if design_function is None else design_function
        self.fit_cache={};self.robust_cache={};self.weights=ds.weights
        self.penalty=np.stack([penalties(c,self.dim) for c in params])
        n=len(params);d=self.dim
        self.G=np.zeros((5,n,d,d));self.B=np.zeros((5,n,d,3));self.Y=np.zeros((5,3));self.N=np.zeros(5)
        # A design can include pre-cue context, so sufficient statistics are
        # accumulated per trial (never grouped while dropping its context).
        for i,e in enumerate(ds.events):
            block=e['block'];h=self.design(params,e,context=context).astype(float)
            w=self.weights[i];x=np.nan_to_num(ds.x[i])
            self.G[block]+=np.matmul(h.transpose(0,2,1),h*w[None,:,None])
            self.B[block]+=np.matmul(h.transpose(0,2,1),w[:,None]*x)
            self.Y[block]+=np.sum(w[:,None]*x*x,axis=0)
            self.N[block]+=1


    def ridge_all(self,blocks):
        ix=list(blocks);count=self.N[ix].sum()
        if count<=0:raise ValueError('Empty training blocks')
        g=self.G[ix].sum(0)/count;b=self.B[ix].sum(0)/count
        scale=np.maximum(np.sqrt(np.diagonal(g,axis1=1,axis2=2)),1e-8)
        gn=g/scale[:,:,None]/scale[:,None,:];bn=b/scale[:,:,None]
        systems=gn[None]+LAMBDAS[:,None,None,None]*np.eye(self.dim)[None,None]*self.penalty[None,:,None,:]
        coef=np.linalg.solve(systems,np.broadcast_to(bn,systems.shape[:2]+(self.dim,3)))
        return coef/scale[None,:,:,None],scale,np.maximum(self.Y[ix].sum(0)/count,1.)

    def validation_errors(self,coef,block,channel_var):
        # lambda, candidate, channel
        err=(np.einsum('lkpc,kpq,lkqc->lkc',coef,self.G[block],coef,optimize=True)
             -2*np.einsum('lkpc,kpc->lkc',coef,self.B[block],optimize=True)+self.Y[block])
        return np.mean(err/channel_var[None,None,:],axis=-1)/self.N[block]

    def matrix(self,indices,candidate_index):
        hs=[];ys=[];ws=[]
        c=self.params[candidate_index]
        for i in indices:
            h=self.design([c],self.ds.events[i],context=self.context)[0].astype(float)
            mask=self.weights[i]>0
            hs.append(h[mask]);ys.append(self.ds.x[i,mask]);ws.append(self.weights[i,mask])
        return np.concatenate(hs),np.concatenate(ys),np.concatenate(ws)/len(indices)

    def robust_fit(self,blocks,ci,lam,iterations=7):
        cache_key=(tuple(sorted(blocks)),ci,lam,iterations)
        if cache_key in self.robust_cache:return self.robust_cache[cache_key]
        idx=np.flatnonzero(np.isin(self.ds.blocks,list(blocks)))
        h,y,w=self.matrix(idx,ci)
        source_scale=np.maximum(np.sqrt(np.sum(w[:,None]*h*h,axis=0)),1e-8)
        channel_scale=np.maximum(np.sqrt(np.sum(w[:,None]*y*y,axis=0)),1.)
        z=h/source_scale;target=y/channel_scale
        gram=z.T@(w[:,None]*z)
        co=solve(gram+lam*np.diag(self.penalty[ci]),z.T@(w[:,None]*target),assume_a='pos')
        # Huber M-estimation on trial/stage-balanced, channel-normalized residuals.
        convergence=[]
        for _ in range(iterations):
            residual=target-z@co;robust=np.minimum(1.,1.5/np.maximum(abs(residual),1e-12))
            new=np.empty_like(co)
            for channel in range(3):
                wc=w*robust[:,channel]
                new[:,channel]=solve(z.T@(wc[:,None]*z)+lam*np.diag(self.penalty[ci]),z.T@(wc*target[:,channel]),assume_a='pos')
            change=float(np.max(abs(new-co)));convergence.append(change);co=new
            if change<1e-6:break
        result=co/source_scale[:,None]*channel_scale[None,:]
        edf=float(np.trace(np.linalg.solve(gram+lam*np.diag(self.penalty[ci]),gram)))
        singular=np.linalg.svd(np.sqrt(w[:,None])*z,compute_uv=False)
        answer=result,dict(source_scale=source_scale,channel_scale=channel_scale,edf=edf,
                          design_singular_values=singular,irls_changes=convergence)
        self.robust_cache[cache_key]=answer
        return answer

    def robust_score(self,coef,ci,held,channel_scale):
        h,y,w=self.matrix(np.flatnonzero(self.ds.blocks==held),ci)
        residual=(y-h@coef)/channel_scale
        loss=np.where(abs(residual)<=1.5,.5*residual**2,1.5*(abs(residual)-.75))
        return float(np.sum(w[:,None]*loss)/3)

    def select(self,blocks,model,shortlist=2,tau_m=None):
        """Every inner screen/refit excludes the outer held block.

        All candidates are screened by nested ridge squared loss. The two best
        candidate/penalty pairs are re-evaluated with exact Huber IRLS in each
        inner fold. This fixed computational rule is part of the estimator.
        """
        blocks=tuple(sorted(blocks));cache_key=(blocks,model,tau_m)
        if cache_key in self.fit_cache:return self.fit_cache[cache_key]
        ids=np.array([i for i,c in enumerate(self.params) if
            (c.model==model or (model in ('N2_fixed_target','N2_equal_penalty') and c.model=='N2'))
            and (model!='N2_fixed_target' or c.target_duration==.2)
            and (model!='N2_equal_penalty' or c.direction_penalty==1.)
            and (tau_m is None or c.tau_m==tau_m)])
        if not len(ids):raise ValueError('No candidates satisfy the profile constraint')
        if len(blocks)<2:raise ValueError('Nested selection needs >=2 blocks')
        scores=np.zeros((len(LAMBDAS),len(self.params)))
        for held in blocks:
            train=[b for b in blocks if b!=held]
            co,_,var=self.ridge_all(train)
            scores+=self.validation_errors(co,held,var)/len(blocks)
        eligible=scores[:,ids]
        order=np.argsort(eligible.ravel(),kind='stable')[:shortlist]
        options=[]
        for flat in order:
            li,local=np.unravel_index(flat,eligible.shape);ci=int(ids[local]);lam=float(LAMBDAS[li]);loss=[]
            for held in blocks:
                co,meta=self.robust_fit([b for b in blocks if b!=held],ci,lam)
                loss.append(self.robust_score(co,ci,held,meta['channel_scale']))
            options.append(dict(candidate_index=ci,lambda_ridge=lam,screen_MSE=float(scores[li,ci]),
                                inner_huber=float(np.mean(loss)),inner_losses=loss))
        best=min(options,key=lambda r:(r['inner_huber'],-r['lambda_ridge'],r['candidate_index']))
        coef,meta=self.robust_fit(blocks,best['candidate_index'],best['lambda_ridge'])
        cand=self.params[best['candidate_index']]
        result=dict(candidate=cand,coef=coef,meta=meta,selection=best,shortlist=options,
             train_blocks=blocks,train_ids=self.ds.ids[np.isin(self.ds.blocks,blocks)].tolist(),
             grid_candidates=len(ids),screening='nested_ridge_all_then_top2_Huber')
        self.fit_cache[cache_key]=result
        return result
