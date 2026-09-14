#!/usr/bin/env python3
"""Assert native parity outputs against frozen reference tensors (stdlib only).

The BF16 transformer uses fused arithmetic and a different reduction order.
Bounds cover observed numerical differences; this is not a same-seed audio test.
"""
import argparse,array,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
limits={'block':.01,'logits':.005,'cached-logits':.005,'velocity':.03,
        'latent-8':.02,'latent-16':.02,'latent-32':.02,'audio':.0001}
report={}
for name in ['text_ids','text-0','text-1','text-2']:
 assert (a.directory/f'{name}.bin').read_bytes()==(a.directory/f'native-{name}.bin').read_bytes(),name
 report[name]={'exact':True}
def floats(path):
 result=array.array('f');result.frombytes(path.read_bytes());return result
for name,limit in limits.items():
 ref='logits' if name=='cached-logits' else name
 x=floats(a.directory/f'{ref}.bin');y=floats(a.directory/f'native-{name}.bin')
 assert len(x)==len(y) and len(x)>0,(name,len(x),len(y))
 assert all(math.isfinite(v) for v in x) and all(math.isfinite(v) for v in y),name
 energy=sum(v*v for v in x);error=sum((v-w)**2 for v,w in zip(x,y))
 relative=math.sqrt(error/energy)
 report[name]={'elements':len(x),'relative_l2':relative,'maximum_error':max(abs(v-w) for v,w in zip(x,y)),'limit':limit}
 assert relative<limit,(name,relative,limit)
# The VAE fixture crosses the 128-frame native tile boundary. Check its seam
# separately so a short local discontinuity cannot hide in a global norm.
x=floats(a.directory/'audio.bin');y=floats(a.directory/'native-audio.bin')
seam=128*1920*2
assert max(abs(v-w) for v,w in zip(x[seam-256:seam+256],y[seam-256:seam+256]))<.0001
(a.directory/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
