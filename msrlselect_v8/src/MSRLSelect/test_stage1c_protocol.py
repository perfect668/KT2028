import sys,torch
sys.path.insert(0,'.')
from stage1c_model import Stage1CDenseModel
from stage1c_train import EarlyStop,lr_for_epoch

def inp():
 torch.manual_seed(3);x=torch.randn(2,4,5,1024);pm=torch.ones(2,4,5,dtype=torch.bool);c=torch.rand(2,4,2);s=torch.rand(2,4);rm=torch.ones(2,4,dtype=torch.bool);return x,pm,c,s,rm

def test_early_stopping_and_scheduler():
 es=EarlyStop(30,20);stops=[]
 for ep in range(1,50):
  _,stop,_=es.update(ep,1. if ep==1 else .5);stops.append((ep,stop))
 assert [e for e,s in stops if s][0]==30
 lr=[lr_for_epoch(e) for e in range(1,121)]
 assert lr[0]<lr[4] and abs(lr[4]-1e-4)<1e-12 and lr[-1]<1e-12 and all(lr[i]>=lr[i+1] for i in range(4,119))

def test_relation_branches_distinct():
 x,pm,c,s,rm=inp();calls=[]
 a=Stage1CDenseModel(spatial=False,relation=True).eval();b=Stage1CDenseModel(spatial=False,relation=False).eval()
 h=a.blocks[0].register_forward_hook(lambda *args:calls.append(1))
 with torch.no_grad():a(x,pm,c,s,rm);b(x,pm,c,s,rm)
 h.remove();assert calls==[1] and len(b.blocks)==0

def test_spatial_gradient_cls_padding_and_effect():
 x,pm,c,s,rm=inp();m=Stage1CDenseModel(spatial=True,relation=True).train();o,aux=m(x,pm,c,s,rm,collect=True);o.sum().backward()
 assert all(b.gate.grad is not None and b.gate.grad.abs().sum()>0 for b in m.blocks)
 assert all(b.bias_mlp[-1].weight.grad is not None and b.bias_mlp[-1].weight.grad.abs().sum()>0 for b in m.blocks)
 rb=aux['blocks'][0]['raw_bias'];assert torch.equal(rb[:,:,0,:],torch.zeros_like(rb[:,:,0,:])) and torch.equal(rb[:,:,:,0],torch.zeros_like(rb[:,:,:,0]))
 m.eval()
 with torch.no_grad():
  y=m(x,pm,c,s,rm);yp=m(x,pm,c.flip(1),s,rm);yb=m(x,pm,c,s,rm,bypass_spatial=True)
 assert not torch.equal(y,yp) and not torch.equal(y,yb)
 rm[:,3]=False;pm[:,3]=False
 with torch.no_grad():
  z1=m(x,pm,c,s,rm);x[:,3]=torch.randn_like(x[:,3])*999;z2=m(x,pm,c,s,rm)
 assert torch.allclose(z1,z2,atol=1e-5)

def test_empty_region_pool_finite():
 x,pm,c,s,rm=inp();pm[:,2]=False;m=Stage1CDenseModel(spatial=False,relation=True)
 assert torch.isfinite(m(x,pm,c,s,rm)).all()
