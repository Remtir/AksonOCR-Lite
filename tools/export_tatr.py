"""Export microsoft/table-transformer-structure-recognition-v1.1-all (MIT) to ONNX.

    python tools/export_tatr.py --model microsoft/table-transformer-structure-recognition-v1.1-all --out weights/tatr.onnx

Input pixel_values [1,3,H,W] (ImageNet-normalized RGB, longest side 1000); outputs logits [1,Q,C+1] and
pred_boxes [1,Q,4] (normalized cx, cy, w, h). Label names are written next to it as tatr_labels.json.
"""
import argparse, json, os

import torch
from transformers import TableTransformerForObjectDetection

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='microsoft/table-transformer-structure-recognition-v1.1-all')
ap.add_argument('--out', required=True); ap.add_argument('--opset', type=int, default=17)
a = ap.parse_args()

model = TableTransformerForObjectDetection.from_pretrained(a.model).eval()


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, pixel_values):
        o = self.m(pixel_values=pixel_values)
        return o.logits, o.pred_boxes


x = torch.rand(1, 3, 800, 1000)
torch.onnx.export(Wrap(model), (x,), a.out, input_names=['pixel_values'], output_names=['logits', 'pred_boxes'],
                  dynamic_axes={'pixel_values': {2: 'h', 3: 'w'}}, opset_version=a.opset, dynamo=False)
json.dump([model.config.id2label[i] for i in range(len(model.config.id2label))],
          open(os.path.join(os.path.dirname(a.out) or '.', 'tatr_labels.json'), 'w'))
print('wrote', a.out)
