import socket,time,json,hashlib
from pathlib import Path
import torch,torchvision
from torchvision.models import ResNet18_Weights
from PIL import Image
socket.setdefaulttimeout(30)
root=Path('/root/autodl-tmp/KT2028');out=root/'environment';weights=ResNet18_Weights.IMAGENET1K_V1
start=time.monotonic();report={'url':weights.url,'download_source':'official PyTorch HTTPS; no third-party weight mirror','scope':'pretrained extractor availability and one image shape check; no prognosis training'}
for attempt in range(1,4):
 try:
  model=torchvision.models.resnet18(weights=weights).eval();break
 except Exception as e:
  report['last_download_error']=type(e).__name__+': '+str(e)[:160]
  if attempt==3:
   report['status']='failed';(out/'resnet18_weights_receipt.json').write_text(json.dumps(report,indent=2));raise
  time.sleep(2*attempt)
feature_extractor=torch.nn.Sequential(*list(model.children())[:-1]).cuda().eval()
image=Image.open(root/'audit/s1_20261009/coarse_access_example/thumbnail.png').convert('RGB')
# This is a software smoke input, not the final regional coarse preprocessing.
x=weights.transforms()(image).unsqueeze(0).cuda()
with torch.inference_mode():feature=feature_extractor(x).flatten(1)
path=Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[-1]
report.update({'status':'verified','path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'output_shape':list(feature.shape),'finite':bool(torch.isfinite(feature).all()),'elapsed_seconds':round(time.monotonic()-start,2),'transform_scope':'default pretrained transform for environment smoke only'})
(out/'resnet18_weights_receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
