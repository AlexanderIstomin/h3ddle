#!/usr/bin/env python3
"""Export development-only YuE2 parity tensors; never needed by the app."""
import argparse,json,os
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--sample',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
import numpy as np
import torch
from yue2 import YuE2Pipeline
from yue2.nar import CachedNAR,song_chunks
from yue2.protocol import SongRequest
pipe=YuE2Pipeline.from_pretrained(str(a.package/'model'),vae=str(a.package/'vae'),device='mps',backend='torch-eager',local_files_only=True,progress=False)
def save(name,t,dtype='<f4'):
 if isinstance(t,torch.Tensor):t=t.detach().float().cpu().numpy()
 np.asarray(t,dtype=dtype).tofile(a.output/(name+'.bin'))
model=pipe._load_model()
text=SongRequest(style='Gentle acoustic folk, warm singing voice, guitar',lyrics='[Verse]\nMorning light across the bay').text()
(a.output/'text.txt').write_text(text)
ids=pipe.tokenizer.encode(text);save('text_ids',ids,'<u4')
for i,variant in enumerate(['café e\u0301 中文\n12345',"We're singing 🎵 <abc>\n\n  test\t",'hello\r\n世界!']):
 (a.output/f'text-{i}.txt').write_text(variant);save(f'text-{i}',pipe.tokenizer.encode(variant),'<u4')
tokens=[151643]+ids+[151847];save('ids',tokens,'<u4')
with torch.inference_mode():
 x=model.model.embed_tokens(torch.tensor([tokens],device='mps'));cos,sin=model.model.rotary_emb(torch.arange(len(tokens),device='mps')[None])
 save('block',model.model.layers[0](x,cos,sin))
 save('logits',model(torch.tensor([tokens],device='mps'),logits_to_keep=1).logits[0,-1])
 prefix=np.load(a.sample/'prefix.npy').tolist();codec=np.load(a.sample/'semantic.npy').tolist()[:25]
 save('prefix',prefix,'<u4');save('codec',codec,'<u4')
 chunk=song_chunks(prefix,codec,831001)[0];engine=CachedNAR(model,chunk)
 save('noise',chunk.noise);save('velocity',engine.velocity(chunk.noise.to(device='mps',dtype=torch.bfloat16),0.0))
 for steps in (8,16,32):
  print('synthesis',steps,flush=True);z=engine.solve(steps);save(f'latent-{steps}',z)
 engine.close()
 # Decode known reference latents independently of synthesis numerical drift.
 z=np.load(a.sample/'latent.npy')[:160];save('decode_input',z)
 print('decode',flush=True);save('audio',pipe.decode(z))
(a.output/'manifest.json').write_text(json.dumps(dict(ids=len(tokens),prefix=len(prefix),frames=len(codec),decode_frames=len(z),version='yue2-infer-0.1.5'),indent=2))
pipe.close();print('Reference export complete',flush=True)
