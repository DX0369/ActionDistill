"""Proper scoring rules, set-hit accuracy, and one shared validation temperature."""
import math
from collections import defaultdict
import torch
import torch.nn.functional as F


def soft_loss(logits, targets, nopts, brier_w=0.0):
    valid = torch.arange(logits.shape[1],device=logits.device)[None,:] < nopts[:,None]
    if targets.shape != logits.shape or not torch.allclose(targets.sum(1),torch.ones(len(targets),device=targets.device)):
        raise ValueError('every target must sum to one')
    if (targets < 0).any() or targets.masked_select(~valid).any():
        raise ValueError('target mass at invalid options or negative target')
    logp = F.log_softmax(logits,dim=1).masked_fill(~valid,0)
    ce = -(targets * logp).sum(1).mean()
    brier = ((F.softmax(logits,dim=1) - targets)**2).sum(1).mean()
    return ce + brier_w*brier, ce, brier


def check_eval_logits(logits):
    # -inf is the upstream padding mask; NaN/+inf are never valid scores.
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise FloatingPointError('NaN or +inf evaluation logits')
    if len(logits) and not torch.isfinite(logits).any(1).all():
        raise FloatingPointError('evaluation row has no finite option logits')


def metrics(logits, targets, datasets, baseline_hits=None, bins=15, *, slices=None, respond_hits=None):
    check_eval_logits(logits)
    definition = f'{bins} equal-width confidence bins on [0,1]; top1 confidence vs accuracy, single-positive-gold rows only; weighted by eligible row count; null if empty.'
    if not len(logits):
        return {'overall':{'count':0,'ece':None,'ece_count':0,'macro_by_dataset':None},'datasets':{},'ece_definition':definition}
    probs = torch.softmax(logits.float(),1)
    confidence, pred = probs.max(1)
    hit = targets.gather(1,pred[:,None]).squeeze(1) > 0
    def one(indices):
        p,t,c,h = probs[indices],targets[indices],confidence[indices],hit[indices]
        logp = F.log_softmax(logits[indices].float(),dim=1).masked_fill(~torch.isfinite(logits[indices]),0)
        single = (t>0).sum(1)==1
        ec,eh = c[single],h[single]
        ece = 0.0 if len(ec) else None
        for b in range(bins):
            sel = (ec >= b/bins) & (ec <= 1 if b==bins-1 else ec < (b+1)/bins)
            if sel.any(): ece += float(sel.float().mean() * (eh[sel].float().mean()-ec[sel].mean()).abs())
        nopts = torch.isfinite(logits[indices]).sum(1)
        result = {'count':len(indices),'acc':float(h.float().mean()),'nll':float(-(t*logp).sum(1).mean()),
                  'brier':float(((p-t)**2).sum(1).mean()),'ece':ece,'ece_count':int(single.sum()),
                  'uniform_random_acc':float(((t>0).sum(1)/nopts).float().mean())}
        if baseline_hits is not None:
            result['most_common_tool_acc'] = sum(baseline_hits[i] for i in indices)/len(indices)
        if respond_hits is not None:
            result['always_respond_acc'] = sum(respond_hits[i] for i in indices)/len(indices)
        for field,labels in (slices or {}).items():
            groups=defaultdict(list)
            for i in indices:groups[labels[i]].append(i)
            result['by_'+field]={label:{'count':len(idx),'acc':float(hit[idx].float().mean())} for label,idx in sorted(groups.items())}
        return result
    by = defaultdict(list)
    for i,d in enumerate(datasets): by[d].append(i)
    result = {'overall':one(list(range(len(datasets)))),'datasets':{d:one(idx) for d,idx in sorted(by.items())},'ece_definition':definition}
    result['overall']['macro_by_dataset'] = sum(v['acc'] for v in result['datasets'].values())/len(by)
    result['macro_by_dataset_definition'] = 'Unweighted mean of dataset set-hit accuracies within this evaluation split.'
    return result


def fit_temperature(logits, targets):
    check_eval_logits(logits)
    if len(logits)==0:
        raise ValueError('validation set is empty; cannot fit temperature')
    valid = torch.isfinite(logits)
    best = (float('inf'),None,None)
    for i in range(120):
        temperature = math.exp(math.log(.3)+(math.log(5)-math.log(.3))*i/119)
        logp = F.log_softmax(logits/temperature,1).masked_fill(~valid,0)
        nll = float(-(targets*logp).sum(1).mean())
        if nll < best[0]: best = nll,temperature,i
    return {'temperature':best[1],'val_soft_nll':best[0],'grid':{'min':.3,'max':5.,'count':120},'at_grid_boundary':best[2] in (0,119)}
