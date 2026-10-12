"""Original UNI only: authorized download, pinned integrity, frozen encoding.

Architecture / normalization follow MahmoodLab/UNI's original model card.
No automatic encoder substitution or random-weight feature generation.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify_checkpoint(path, spec):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Original UNI checkpoint missing; obtain official model access first')
    if path.stat().st_size != spec['size_bytes']:
        raise ValueError('Original UNI checkpoint size mismatch')
    digest = sha256_file(path)
    if digest != spec['sha256']:
        raise ValueError('Original UNI checkpoint SHA256 does not match pinned official LFS metadata')
    return digest


def download_checkpoint(directory, spec):
    from huggingface_hub import get_token, hf_hub_download
    if not get_token():
        raise PermissionError('Log in to an account with approved MahmoodLab/UNI access; no token is printed or stored in project files')
    # Do not replace official authenticated access with an unauthenticated mirror.
    path = hf_hub_download(repo_id=spec['repo_id'], filename=spec['filename'],
                           revision=spec['revision'], local_dir=str(directory),
                           endpoint='https://huggingface.co', token=True)
    verify_checkpoint(path, spec)
    return Path(path)


class OriginalUNI:
    def __init__(self, checkpoint, spec, device='cuda'):
        self.checkpoint_sha256 = verify_checkpoint(checkpoint, spec)
        import torch
        import timm
        from torchvision import transforms
        self.torch = torch
        self.device = torch.device(device)
        self.model = timm.create_model(spec['model_name'], img_size=224, patch_size=16,
                                      init_values=spec['init_values'], num_classes=0,
                                      dynamic_img_size=True, pretrained=False)
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        del weights
        self.model.requires_grad_(False).eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(spec['mean'], spec['std'])])

    def encode(self, images):
        import numpy as np
        if not images or any(i.mode != 'RGB' or i.size != (224, 224) for i in images):
            raise ValueError('UNI input must be nonempty, already sampled RGB 224x224 patches')
        x = self.torch.stack([self.transform(i) for i in images]).to(self.device)
        with self.torch.inference_mode():
            features = self.model(x).cpu().float().numpy()
        if features.shape != (len(images), 1024) or not np.isfinite(features).all():
            raise ValueError('Original UNI output contract mismatch')
        return features


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--download-dir', type=Path)
    ap.add_argument('--image', type=Path, help='Optional real 224px patch for frozen forward verification')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--receipt', type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.config.read_text())['uni']
    if args.receipt.exists():
        raise FileExistsError('Receipt already exists; preserve earlier evidence')
    if bool(args.checkpoint) == bool(args.download_dir):
        raise ValueError('Choose one existing checkpoint or authorized download directory')
    checkpoint = args.checkpoint or download_checkpoint(args.download_dir, spec)
    digest = verify_checkpoint(checkpoint, spec)
    receipt = {'status': 'weight_verified', 'repo_id': spec['repo_id'],
               'revision': spec['revision'], 'weight_sha256': digest,
               'weight_bytes': checkpoint.stat().st_size, 'trained': False}
    if args.image:
        from PIL import Image
        encoder = OriginalUNI(checkpoint, spec, args.device)
        features = encoder.encode([Image.open(args.image).convert('RGB')])
        receipt.update(status='weight_and_real_forward_verified', shape=list(features.shape),
                       image_sha256=sha256_file(args.image), device=args.device)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
