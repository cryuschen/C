"""Blocked nested selection, sufficient-statistic screening, robust refitting."""
from collections import OrderedDict
import numpy as np
from scipy.linalg import solve
from .data import TIME, STAGES, stage_masks, sample_weights
from .model import design_bank, stimulus_key, predict

LAMBDAS=np.array([.001,.01,.1,1.,10.])


class Bank:
    def __init__(self,ds,params):
        self.ds=ds;self.params=params;self.cache=OrderedDict();self.fit_cache={}
        self.single_cache=OrderedDict();self.robust_cache={};self.weights=ds.weights
        n=len(params);self.G=np.zeros((5,n,9,9));self.B=np.zeros((5,n,9,3));self.Y=np.zeros((5,3));self.N=np.zeros(5)
        keys={stimulus_key(e) for e in ds.events}
        weights=self.weights;x=np.nan_to_num(ds.x)
        for key in sorted(keys):
            idx=np.array([i for i,e in enumerate(ds.events) if stimulus_key(e)==key])
            h=self.design(ds.events[idx[0]]).astype(float)
            for block in np.unique(ds.blocks[idx]):
                ii=idx[ds.blocks[idx]==block];w=weights[ii]
                # Sum over trials before multiplying templates: exact sufficient statistics.
                gw=w.sum(0);by=np.einsum('nt,ntc->tc',w,x[ii])
                self.G[block]+=np.matmul(h.transpose(0,2,1),h*gw[None,:,None])
                self.B[block]+=np.matmul(h.transpose(0,2,1),by)
                self.Y[block]+=np.einsum('nt,ntc,ntc->c',w,x[ii],x[ii])
                self.N[block]+=len(ii)
        self.cache.clear()

    def design(self,event):
        key=stimulus_key(event)
        if key not in self.cache:
            self.cache[key]=design_bank(self.params,event,self.ds.mode)
            if len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def ridge_all(self,blocks):
        ix=list(blocks);count=self.N[ix].sum()
        if count<=0:raise ValueError('Empty training blocks')
        g=self.G[ix].sum(0)/count;b=self.B[ix].sum(0)/count
        scale=np.maximum(np.sqrt(np.diagonal(g,axis1=1,axis2=2)),1e-8)
        gn=g/scale[:,:,None]/scale[:,None,:];bn=b/scale[:,:,None]
        systems=gn[None]+LAMBDAS[:,None,None,None]*np.eye(9)[None,None]
        coef=np.linalg.solve(systems,np.broadcast_to(bn,systems.shape[:2]+(9,3)))
        return coef/scale[None,:,:,None],scale,np.maximum(self.Y[ix].sum(0)/count,1.)

    def validation_errors(self,coef,block,channel_var):
        # lambda, candidate, channel
        err=(np.einsum('lkpc,kpq,lkqc->lkc',coef,self.G[block],coef,optimize=True)
             -2*np.einsum('lkpc,kpc->lkc',coef,self.B[block],optimize=True)+self.Y[block])
        return np.mean(err/channel_var[None,None,:],axis=-1)/self.N[block]

    def matrix(self,indices,candidate_index):
        hs=[];ys=[];ws=[]
        # Single-candidate computation avoids keeping the entire grid in RAM.
        c=self.params[candidate_index]
        for i in indices:
            e=self.ds.events[i];key=(candidate_index,stimulus_key(e))
            if key not in self.single_cache:
                self.single_cache[key]=design_bank([c],e,self.ds.mode)[0].astype(float)
                if len(self.single_cache)>256:self.single_cache.popitem(last=False)
            self.single_cache.move_to_end(key)
            mask=self.weights[i]>0
            hs.append(self.single_cache[key][mask]);ys.append(self.ds.x[i,mask]);ws.append(self.weights[i,mask])
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
        co=solve(gram+lam*np.eye(9),z.T@(w[:,None]*target),assume_a='pos')
        # Huber M-estimation on trial/stage-balanced, channel-normalized residuals.
        convergence=[]
        for _ in range(iterations):
            residual=target-z@co;robust=np.minimum(1.,1.5/np.maximum(abs(residual),1e-12))
            new=np.empty_like(co)
            for channel in range(3):
                wc=w*robust[:,channel]
                new[:,channel]=solve(z.T@(wc[:,None]*z)+lam*np.eye(9),z.T@(wc*target[:,channel]),assume_a='pos')
            change=float(np.max(abs(new-co)));convergence.append(change);co=new
            if change<1e-6:break
        result=co/source_scale[:,None]*channel_scale[None,:]
        edf=float(np.trace(np.linalg.solve(gram+lam*np.eye(9),gram)))
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

    def select(self,blocks,model,shortlist=2):
        """Every inner screen/refit excludes the outer held block.

        All candidates are screened by nested ridge squared loss. The two best
        candidate/penalty pairs are re-evaluated with exact Huber IRLS in each
        inner fold. This fixed computational rule is part of the estimator.
        """
        blocks=tuple(sorted(blocks));cache_key=(blocks,model)
        if cache_key in self.fit_cache:return self.fit_cache[cache_key]
        ids=np.array([i for i,c in enumerate(self.params) if c.model==model])
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


def stage_baselines(ds,train):
    result=[]
    for stage in range(3):
        values=[]
        for i in np.flatnonzero(train):
            e=ds.events[i];mask=stage_masks(e['target_s'],e['end_s'])[stage]
            if mask.any():values.append(ds.x[i,mask].mean(0))
        result.append(np.mean(values,axis=0) if values else np.zeros(3))
    return np.array(result)


def evaluate(bank,models,progress=None):
    ds=bank.ds;rows=[];choices=[];predictions={m:np.full_like(ds.x,np.nan) for m in models}
    fits={}
    for held in sorted(np.unique(ds.blocks)):
        train=ds.blocks!=held;baseline=stage_baselines(ds,train)
        for model in models:
            result=bank.select(sorted(set(ds.blocks[train])),model);fits[(int(held),model)]=result
            cand=result['candidate'];coef=result['coef'];scale=result['meta']['channel_scale']
            choices.append(dict(dataset=ds.key,held_block=int(held),model=model,parameters=cand.record(),
                  **{k:v for k,v in result.items() if k not in ('candidate','coef')},coef=coef))
            for i in np.flatnonzero(ds.blocks==held):
                e=ds.events[i];pred=predict(e,cand,coef,ds.mode);predictions[model][i]=pred
                masks=stage_masks(e['target_s'],e['end_s'])
                for stage,mask in enumerate(masks):
                    if not mask.any():continue
                    for ch in range(3):
                        yy=ds.x[i,mask,ch];pp=pred[mask,ch]
                        rows.append(dict(dataset=ds.key,trial_id=e['trial_id'],block=int(held),cue=e['cue'],
                            model=model,stage=STAGES[stage],channel=ch,n_samples=int(mask.sum()),
                            MSE=float(np.mean((yy-pp)**2)),zero_MSE=float(np.mean(yy**2)),
                            baseline_MSE=float(np.mean((yy-baseline[stage,ch])**2)),
                            normalized_MSE=float(np.mean(((yy-pp)/scale[ch])**2))))
        if progress:progress(f'{ds.key}: outer block {held+1}/5 completed')
    return rows,choices,predictions,fits
