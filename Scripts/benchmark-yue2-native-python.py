#!/usr/bin/env python3
"""Fixed-work Python baseline for the native YuE2 benchmark harness.

Development only. Use the pinned yue2-infer 0.1.5 runtime. Both backends replay
identical tokens/noise/latents; generated song lengths and RNGs cannot skew cost.
"""
import argparse,json,os,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--sample',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=750);p.add_argument('--replay',action='store_true',help='Replay all score and music tokens, keeping stage cache behavior');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
import numpy as np
import torch
from yue2 import YuE2Pipeline
from yue2.nar import CachedNAR,song_chunks
from yue2.modeling_yue2 import StaticKVCache
report={'backend':'python','frames':a.frames,'trials':[],'metal_validation':os.environ.get('MTL_DEBUG_LAYER','default')}
def timed(name,fn):
 torch.mps.synchronize();start=time.perf_counter();result=fn();torch.mps.synchronize()
 seconds=time.perf_counter()-start;report['trials'].append({'stage':name,'seconds':seconds});print(name,seconds,flush=True)
 (a.output/'python-timing.json').write_text(json.dumps(report,indent=2)+'\n');return result
start=time.perf_counter()
pipe=YuE2Pipeline.from_pretrained(str(a.package/'model'),vae=str(a.package/'vae'),device='mps',backend='torch-eager',local_files_only=True,progress=False)
model=pipe._load_model();torch.mps.synchronize();report['load_seconds']=time.perf_counter()-start
prefix=np.load(a.sample/'prefix.npy').tolist();codec=np.load(a.sample/'semantic.npy').tolist()[:a.frames]
if len(codec)!=a.frames:raise ValueError('Sample is shorter than requested workload')
chunk=song_chunks(prefix,codec,831001)[0]
for name,values,dtype in [('prefix',prefix,'<u4'),('codec',codec,'<u4'),('noise',chunk.noise.numpy(),'<f4'),('decode_input',np.load(a.sample/'latent.npy')[:a.frames],'<f4')]:np.asarray(values,dtype=dtype).tofile(a.output/(name+'.bin'))
(a.output/'benchmark-manifest.json').write_text(json.dumps(dict(frames=a.frames,prefix=len(prefix),decode_tokens=a.frames if a.replay else 128,steps=[8,16,32],mode='pipeline-replay' if a.replay else 'cold-stages')))
with torch.inference_mode():
 if a.replay:
  prompt=prefix.index(151847)+1
  assert prefix[-2:]==[151848,151851]
  def new_cache(capacity):
   return StaticKVCache(num_layers=28,batch_size=1,num_kv_heads=8,max_seq_len=capacity,head_dim=128,dtype=torch.bfloat16,device='mps')
  def replay_plan():
   cache=new_cache(prompt+4096)
   model(torch.tensor([prefix[:prompt]],device='mps'),past_key_values=cache,use_cache=True,logits_to_keep=1)
   torch.mps.synchronize()
   for token in prefix[prompt:-2]:
    model(torch.tensor([[token]],device='mps'),past_key_values=cache,use_cache=True,logits_to_keep=1)
    torch.mps.synchronize()
  timed('plan',replay_plan)
  def replay_semantic():
   # The pinned pipeline starts a fresh cache at this stage.
   cache=new_cache(len(prefix)+a.frames)
   model(torch.tensor([prefix],device='mps'),past_key_values=cache,use_cache=True,logits_to_keep=1)
   torch.mps.synchronize()
   for token in codec[:-1]:
    model(torch.tensor([[token+151853]],device='mps'),past_key_values=cache,use_cache=True,logits_to_keep=1)
    torch.mps.synchronize()
  timed('semantic',replay_semantic)
 for trial in range(0 if a.replay else 2):
  cache=StaticKVCache(num_layers=28,batch_size=1,num_kv_heads=8,max_seq_len=len(prefix)+128,head_dim=128,dtype=torch.bfloat16,device='mps')
  ids=torch.tensor([prefix],device='mps')
  timed(f'prefill-{trial}',lambda:model(ids,past_key_values=cache,use_cache=True,logits_to_keep=1))
  def decode_tokens():
   for token in codec[:128]:model(torch.tensor([[token+151853]],device='mps'),past_key_values=cache,use_cache=True,logits_to_keep=1)
  timed(f'ar128-{trial}',decode_tokens);del cache
 for steps in (8,16,32):
  def solve():
   engine=CachedNAR(model,chunk)
   try:return engine.solve(steps).float().cpu().numpy()
   finally:engine.close()
  result=timed(f'synthesis-{steps}',solve)
  result.astype('<f4').tofile(a.output/f'python-latent-{steps}.bin')
 z=np.load(a.sample/'latent.npy')[:a.frames]
 audio=timed('decode',lambda:pipe.decode(z));np.asarray(audio,dtype='<f4').tofile(a.output/'python-audio.bin')
pipe.close()
