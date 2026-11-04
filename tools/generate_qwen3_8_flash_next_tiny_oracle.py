#!/usr/bin/env python3
"""Generate small BF16 HC/QSA/PLE/DeltaNet fixtures from upstream PyTorch modules."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from generate_qwen3_8_flash_next_oracle import reference_namespace, Cache


def init(module):
    with torch.no_grad():
        for i, (name, p) in enumerate(module.named_parameters()):
            values=torch.sin(torch.arange(p.numel()).float()*.37+i)*.35
            if 'norm.weight' in name and 'hc_norm' not in name:
                values*=.2
            p.copy_(values.reshape(p.shape))
    return module.bfloat16()


def record(module):
    return {k:{'shape':list(v.shape),'values':v.flatten().float().tolist()} for k,v in module.state_dict().items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-dir',type=Path)
    a=p.parse_args(); torch.set_num_threads(1)
    ns=reference_namespace(a.reference_source)
    c=SimpleNamespace(hidden_size=4,hc_count=2,hc_lowrank=3,rms_norm_eps=1e-6,
        head_dim=8,num_attention_heads=2,num_key_value_heads=1,attention_dropout=0.,attention_bias=False,
        indexer_n_heads=2,indexer_kv_heads=1,indexer_head_dim=8,indexer_budget=4,indexer_compress_ratio=2,
        _attn_implementation='eager',linear_num_key_heads=1,linear_num_value_heads=2,
        linear_key_head_dim=4,linear_value_head_dim=4,linear_conv_kernel_dim=3,hidden_act='silu',
        output_gate_type='sigmoid',layer_types=['linear_attention'],ple_embed_dim=4,ngram_size=3,ple_conv_kernel_size=4)
    data={'generator':Path(__file__).name,'upstream_source_sha256':hashlib.sha256(a.reference_source.read_bytes()).hexdigest()}
    with torch.inference_mode():
        hc=init(ns['Qwen4ExpTextGatedResidual'](c))
        inputs=torch.cos(torch.arange(40).float()*.19).view(5,8).bfloat16()
        mixed,original,injection=hc(inputs)
        data['hc']={'weights':record(hc),'inputs':inputs.float().tolist(),'mixed':mixed.float().tolist(),'injection':injection.float().tolist()}
        qsa=init(ns['Qwen4ExpTextAttention'](c,0)); cache=Cache()
        inputs=torch.sin(torch.arange(52).float()*.23).view(13,4).bfloat16(); outputs=[]
        for pos,x in enumerate(inputs):
            freqs=torch.arange(pos+1).float()[:,None]/(10000.**(torch.arange(0,4,2).float()/4))
            cos=torch.cat((freqs.cos(),freqs.cos()),-1)[None].bfloat16(); sin=torch.cat((freqs.sin(),freqs.sin()),-1)[None].bfloat16()
            y,_=qsa(x.view(1,1,-1),(cos,sin),torch.zeros(1,1,1,pos+1),cache)
            outputs.append(y.flatten().float().tolist())
        data['qsa']={'weights':record(qsa),'inputs':inputs.float().tolist(),'outputs':outputs}
        delta=init(ns['Qwen4ExpTextGatedDeltaNet'](c,0)); cache=Cache(); outputs=[]
        for x in inputs:
            y=delta(x.view(1,1,-1),cache_params=cache); outputs.append(y.flatten().float().tolist())
        data['delta']={'weights':record(delta),'inputs':inputs.float().tolist(),'outputs':outputs}
        ple=init(ns['Qwen4ExpTextPLELayer'](c,0,0)); cache=Cache()
        class Embedding(torch.nn.Module):
            def forward(self, ids, cache):
                return inputs[ids.item()].view(1,1,-1)
        ple.ple_embedding=Embedding()
        hidden=torch.cos(torch.arange(104).float()*.11).view(13,8).bfloat16(); outputs=[]
        for i,x in enumerate(hidden):
            outputs.append(ple(x.view(1,1,-1),torch.tensor([[i]]),cache).flatten().float().tolist())
        data['ple']={'weights':record(ple),'inputs':hidden.float().tolist(),'embeddings':inputs.float().tolist(),'outputs':outputs}
    if a.model_dir:
        import jinja2
        from tokenizers import Tokenizer
        template_text=(a.model_dir/'chat_template.jinja').read_text()
        template=jinja2.Environment().from_string(template_text)
        tokenizer=Tokenizer.from_file(str(a.model_dir/'tokenizer.json'))
        messages=[{'role':'system','content':'  Be concise.  '},{'role':'user','content':'  你好，世界 🌍  '},
                  {'role':'assistant','content':' 回答 ','reasoning_content':'  历史推理  '},{'role':'user','content':'继续'}]
        data['template_sha256']=hashlib.sha256(template_text.encode()).hexdigest()
        data['prompts']=[]
        for thinking in (True,False):
            for preserve in (True,False):
                for effort in ('xhigh','medium','low'):
                    options=dict(enable_thinking=thinking,preserve_thinking=preserve,reasoning_effort=effort,add_generation_prompt=True)
                    rendered=template.render(messages=messages,**options)
                    data['prompts'].append(dict(messages=messages,options=options,prompt=rendered,token_ids=tokenizer.encode(rendered).ids))
    a.output.write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
