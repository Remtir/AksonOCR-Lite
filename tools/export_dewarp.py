"""Export the MonkeyOCRv2 document preprocessor (Apache-2.0) to ONNX: U2NETP page mask + ViTEraser-style warp net.

    python tools/export_dewarp.py --model-dir <MonkeyOCRv2 dir with preprocessor1/2.pth> \
        --code <dir containing modeling_preprocessor.py> --out weights/

Writes dewarp_mask.onnx (x [1,3,512,512] RGB in [0,1] -> mask [1,1,512,512]) and
dewarp_grid.onnx (masked x [1,3,512,512] -> sampling grid [1,2,h,w] in [-1,1], align_corners=True).
"""
import argparse, os, sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--model-dir', required=True); ap.add_argument('--code', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--opset', type=int, default=17)
a = ap.parse_args()
sys.path.insert(0, a.code)
from modeling_preprocessor import Preprocessor  # noqa: E402

pre = Preprocessor(a.model_dir, device='cpu', batch_size=1)


class Mask(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        out = self.m(x)
        return out[0] if isinstance(out, (tuple, list)) else out


class Grid(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m(x)[0]


x = torch.rand(1, 3, 512, 512)
with torch.no_grad():
    print('mask out', [t.shape for t in (pre.model1(x) if isinstance(pre.model1(x), (tuple, list)) else [pre.model1(x)])][:2])
    print('grid out', [t.shape for t in pre.model2(x)])
os.makedirs(a.out, exist_ok=True)
torch.onnx.export(Mask(pre.model1), (x,), os.path.join(a.out, 'dewarp_mask.onnx'), input_names=['x'], output_names=['mask'],
                  opset_version=a.opset, dynamo=False)
# the Swin warp net trips the TorchScript exporter's shape inference; the dynamo exporter handles it
prog = torch.onnx.export(Grid(pre.model2), (x,), input_names=['x'], output_names=['grid'], opset_version=18, dynamo=True,
                         external_data=False)
prog.save(os.path.join(a.out, 'dewarp_grid.onnx'))
print('wrote', a.out)
