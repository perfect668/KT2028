"""Executable adapter checks against downloaded official models (server only)."""
import argparse,copy,importlib.util,math,pathlib,sys
import numpy as np
import torch
from kt2028.open_survival_adapter import cox_breslow, baseline_curves, risk_metrics, make_model, load_file

def main(root,method):
    torch.set_num_threads(2);torch.manual_seed(41)
    # Independent hand calculation, risk direction, tied-event/censor order invariance.
    r=torch.tensor([.2,-.4,.5,.8],dtype=torch.float64,requires_grad=True)
    t=torch.tensor([2.,2.,3.,4.]);e=torch.tensor([1.,1.,0.,1.])
    actual=cox_breslow(r,t,e)
    expected=(2*torch.logsumexp(r,0)-r[0]-r[1])/3
    torch.testing.assert_close(actual,expected)
    for perm in [torch.tensor([1,0,2,3]),torch.tensor([3,2,1,0])]:
        torch.testing.assert_close(actual,cox_breslow(r[perm],t[perm],e[perm]))
    cox_breslow(r,t,torch.zeros_like(e)).backward();assert torch.equal(r.grad,torch.zeros_like(r))
    native=load_file('nn_loss_check',root/'official/nnMIL/training/losses/survival_loss.py').SurvivalLoss()
    torch.testing.assert_close(actual,native(r,e,t),atol=1e-9,rtol=1e-9)
    rows=[dict(cohort='tcga',survival_target=dict(time_days=float(a),event=int(b))) for a,b in zip(t,e)]
    s=baseline_curves(np.zeros(4),rows,np.array([0.,1.]),[0,2,4])
    np.testing.assert_allclose(s[0],[math.exp(-.5),math.exp(-1.5)])
    assert (s[1]<s[0]).all()
    assert risk_metrics([4,3,2,1],[dict(survival_target=dict(time_days=i+1,event=1)) for i in range(4)])['c_index']==1
    print('Cox ties/native agreement, censoring, baseline survival, risk direction: passed',flush=True)
    model=make_model(method,root/'official')
    if method=='nnMIL':
        x=torch.randn(1,7,1024);mask=torch.tensor([[1,1,1,1,0,0,0]],dtype=torch.bool)
        model.eval();a=model(x[:,:4],is_cox=True)['logits'];b=model(x,valid_mask=mask,is_cox=True)['logits']
        torch.testing.assert_close(a,b)
        from kt2028.run_open_survival import predict_one
        model.train();clone=copy.deepcopy(model);state=torch.get_rng_state()
        y=model(x[:,:4],is_cox=True)['logits'].sum();y.backward()
        torch.set_rng_state(state)
        z=predict_one(clone,method,x[0,:4],torch.device('cpu'),0,1,True);z.backward()
        torch.testing.assert_close(y,z)
        for p,q in zip(model.parameters(),clone.parameters()):torch.testing.assert_close(p.grad,q.grad)
        print('Official masked padding and checkpoint output/gradient/dropout equality: passed',flush=True)
    elif method=='OTSurv':
        from kt2028.run_open_survival import predict_one
        x=torch.randn(19,1024);clone=copy.deepcopy(model);model.train();clone.train();state=torch.get_rng_state()
        y=model.forward_no_loss([x,2,0])['logits'].sum();y.backward();torch.set_rng_state(state)
        z=predict_one(clone,method,x,torch.device('cpu'),0,2,True);z.backward()
        torch.testing.assert_close(y,z)
        for p,q in zip(model.parameters(),clone.parameters()):torch.testing.assert_close(p.grad,q.grad)
        print('Official differentiable OT checkpoint output/gradient/dropout equality: passed',flush=True)
    else:
        from kt2028.open_survival_adapter import make_panther
        path=root/'test_prototypes.npy';np.save(path,np.random.default_rng(41).normal(size=(16,1024)).astype('float32'))
        enc=make_panther(root/'official',path).eval();x=torch.randn(1,23,1024)
        a,_=enc(x);b,_=enc(x[:,torch.randperm(23)])
        torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)
        assert not any(p.requires_grad for p in enc.parameters())
        torch.testing.assert_close(model.forward_no_loss(a)['logits'],a@model.classifier.weight.T)
        print('Official GMM permutation invariance/frozen prototypes/native linear head: passed',flush=True)
    print('ALL CHECKS PASSED',method,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--method',required=True)
    a=p.parse_args();main(a.root,a.method)
