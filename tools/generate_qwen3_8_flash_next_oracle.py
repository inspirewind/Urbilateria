#!/usr/bin/env python3
"""Stream a Flash-Next PyTorch oracle using unmodified upstream mathematical modules.

--reference-source is modeling_qwen4_exp.py from huggingface/transformers. Only
AST-selected definitions are executed; framework decorators are removed. Checkpoint
storage, cache plumbing and split PLE embedding lookup are replaced for bounded RAM.
No Rust operations are used. The artifact records the upstream source SHA256.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open


def reference_namespace(path):
    wanted = {'Qwen4ExpTextRMSNorm', 'Qwen4ExpTextRMSNormGated', 'apply_mask_to_padding_states',
              'causal_conv1d_update', 'causal_conv1d_fn', 'l2norm', 'torch_chunk_gated_delta_rule',
              'torch_recurrent_gated_delta_rule', 'Qwen4ExpTextGatedDeltaNet', 'rotate_half',
              'apply_rotary_pos_emb', 'Qwen4ExpTextQSAIndexer', 'repeat_kv', 'eager_attention_forward',
              'Qwen4ExpTextAttention', 'Qwen4ExpTextMLP', 'Qwen4ExpTextExperts', 'Qwen4ExpTextTopKRouter',
              'Qwen4ExpTextSparseMoeBlock', 'Qwen4ExpTextGatedResidual', 'Qwen4ExpTextPLELayer',
              'Qwen4ExpTextDecoderLayer'}
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef)):
                child.decorator_list = []
    class Placeholder(nn.Module):
        def __init__(self, *args):
            super().__init__()
    namespace = dict(is_torchdynamo_exporting=lambda: False, torch=torch, nn=nn, F=F, math=math, GradientCheckpointingLayer=nn.Module,
                     ACT2FN={'silu': F.silu, 'sigmoid': torch.sigmoid},
                     Qwen4ExpTextNGramEmbedding=Placeholder,
                     ALL_ATTENTION_FUNCTIONS=SimpleNamespace(get_interface=lambda name, default: default))
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace


class Checkpoint:
    def __init__(self, directory):
        self.directory = directory
        self.mapping = json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
        self.handles = {}
    def slice(self, name):
        shard = self.mapping[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(self.directory/shard, framework='pt', device='cpu')
        return self.handles[shard].get_slice(name)
    def tensor(self, name):
        return self.slice(name)[:]


class Cache:
    def __init__(self):
        self.conv, self.recurrent, self.index, self.kv = {}, {}, {}, {}
    def get_conv_state(self, layer, state_idx=0):
        return self.conv.get((layer, state_idx))
    def update_conv_state(self, x, layer, state_idx=0, conv_kernel_size=4):
        old = self.conv.get((layer,state_idx))
        full = torch.cat((old,x),dim=-1) if old is not None else x
        self.conv[layer,state_idx] = F.pad(full,(max(0,conv_kernel_size-full.shape[-1]),0))[...,-conv_kernel_size:].clone()
        return full
    def is_recording_past(self, layer):
        return False
    def get_recurrent_state(self, layer, state_idx=0):
        return self.recurrent.get(layer)
    def update_recurrent_state(self, x, layer):
        self.recurrent[layer] = x
    def update_indexer(self, x, layer):
        self.index[layer] = torch.cat((self.index[layer],x),dim=1) if layer in self.index else x
        return self.index[layer]
    def update(self,k,v,layer):
        if layer in self.kv:
            oldk,oldv=self.kv[layer]
            k,v=torch.cat((oldk,k),dim=-2),torch.cat((oldv,v),dim=-2)
        self.kv[layer]=(k,v)
        return k,v


class StreamedExperts(nn.Module):
    def __init__(self, checkpoint, prefix):
        super().__init__(); self.checkpoint,self.prefix=checkpoint,prefix
    def forward(self, hidden, indices, weights):
        output=torch.zeros_like(hidden)
        for expert in sorted(set(indices.flatten().tolist())):
            tokens,slots=torch.where(indices==expert)
            gate,up=F.linear(hidden[tokens],self.checkpoint.slice(self.prefix+'.gate_up_proj')[expert]).chunk(2,dim=-1)
            values=F.linear(F.silu(gate)*up,self.checkpoint.slice(self.prefix+'.down_proj')[expert])
            output.index_add_(0,tokens,(values*weights[tokens,slots,None]).to(output.dtype))
        return output


class StreamedNGram(nn.Module):
    def __init__(self, checkpoint, prefix, config):
        super().__init__(); self.checkpoint,self.prefix,self.config=checkpoint,prefix,config
        self.history=[]
        self.multipliers=checkpoint.tensor(prefix+'.layer_multipliers')
        self.sizes=checkpoint.tensor(prefix+'.ngram_heads_vocab_sizes')
        self.offsets=checkpoint.tensor(prefix+'.ngram_heads_offsets')
        self.rows=checkpoint.slice(prefix+'.ngram_embedding.shard_0.weight').get_shape()[0]
    def forward(self, ids, cache):
        assert ids.numel()==1
        token=ids.item(); c=self.config
        shifted=[token]+[self.history[-i] if len(self.history)>=i else c.eos_token_id for i in range(1,c.ngram_size)]
        mixed=torch.tensor(shifted,dtype=torch.int64)*self.multipliers
        blocks=[]
        value=mixed[0]
        for lag in range(1,c.ngram_size):
            value=value^mixed[lag]
            span=slice((lag-1)*c.heads_per_ngram,lag*c.heads_per_ngram)
            blocks.append(value.remainder(self.sizes[span])+self.offsets[span])
        rows=torch.cat(blocks).tolist()
        self.last_ids=rows
        values=torch.cat([self.checkpoint.slice(f'{self.prefix}.ngram_embedding.shard_{row//self.rows}.weight')[row%self.rows] for row in rows])
        self.history=[] if token==c.eos_token_id else (self.history+[token])[-c.ngram_size+1:]
        return values.view(1,1,-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir',type=Path)
    parser.add_argument('--reference-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokens',type=int,nargs='+',default=[9419,11])
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    ns=reference_namespace(args.reference_source)
    config=SimpleNamespace(**json.loads((args.model_dir/'config.json').read_text())['text_config'])
    config.norm_topk_prob=True; config._attn_implementation='eager'; config.seed=0
    checkpoint=Checkpoint(args.model_dir); cache=Cache(); steps=[]; ple_modules={}
    with torch.inference_mode():
        for position,token in enumerate(args.tokens):
            hidden=checkpoint.slice('model.language_model.embed_tokens.weight')[token].view(1,1,-1).repeat(1,1,config.hc_count)
            rotary=int(config.head_dim*config.partial_rotary_factor)
            freqs=torch.arange(position+1).float()[:,None]/(config.rope_parameters['rope_theta']**(torch.arange(0,rotary,2).float()/rotary))
            cosine=torch.cat((freqs.cos(),freqs.cos()),dim=-1)[None].bfloat16()
            sine=torch.cat((freqs.sin(),freqs.sin()),dim=-1)[None].bfloat16()
            traces=[]
            for layer_id in range(config.num_hidden_layers):
                prefix=f'model.language_model.layers.{layer_id}.'
                with torch.device('meta'):
                    layer=ns['Qwen4ExpTextDecoderLayer'](config,layer_id)
                weights={name[len(prefix):]:checkpoint.tensor(name) for name in checkpoint.mapping
                         if name.startswith(prefix) and '.mlp.experts.' not in name and '.ple.ple_embedding.' not in name}
                missing,unexpected=layer.load_state_dict(weights,strict=False,assign=True)
                assert set(missing)=={'mlp.experts.gate_up_proj','mlp.experts.down_proj'},(missing,unexpected)
                assert not unexpected
                layer.mlp.experts=StreamedExperts(checkpoint,prefix+'mlp.experts')
                if layer.ple is not None:
                    if layer_id not in ple_modules:
                        ple_modules[layer_id]=StreamedNGram(checkpoint,prefix+'ple.ple_embedding',config)
                    layer.ple.ple_embedding=ple_modules[layer_id]
                routed=[]
                layer.mlp.gate.register_forward_hook(lambda module,inputs,output:routed.extend(output[2].flatten().tolist()))
                hidden=layer(hidden,position_embeddings=(cosine,sine),attention_mask=torch.zeros(1,1,1,position+1),
                             past_key_values=cache,ple_input_ids=torch.tensor([[token]]))
                flat=hidden.flatten().float()
                traces.append({'first_16':flat[:16].tolist(),'l2_norm':flat.norm().item(),'experts':routed})
                print(f'position={position} layer={layer_id+1}/48',flush=True)
                del layer,weights
            prefix='model.language_model.hyper_connection_mixer.'
            with torch.device('meta'):
                mixer=ns['Qwen4ExpTextGatedResidual'](config,use_combine=False)
            mixer.load_state_dict({name[len(prefix):]:checkpoint.tensor(name) for name in checkpoint.mapping if name.startswith(prefix)},assign=True)
            final=mixer(hidden).flatten()
            logits=torch.cat([F.linear(final,checkpoint.slice('lm_head.weight')[i:i+1024]) for i in range(0,config.vocab_size,1024)]).float()
            top_values,top_ids=logits.topk(20)
            samples=sorted(set(range(0,config.vocab_size,251))|set(top_ids.tolist()))
            steps.append({'token':token,'argmax':logits.argmax().item(),'layers':traces,
                          'ple_ids':ple_modules[1].last_ids,
                          'top_20':list(zip(top_ids.tolist(),top_values.tolist())),
                          'logit_samples':[[i,logits[i].item()] for i in samples]})
    args.output.write_text(json.dumps({'checkpoint':'Qwen3.8-Flash-Next','torch_version':torch.__version__,
        'generator':Path(__file__).name,'reference_source':'huggingface/transformers/models/qwen4_exp/modeling_qwen4_exp.py',
        'upstream_source_sha256':hashlib.sha256(args.reference_source.read_bytes()).hexdigest(),'steps':steps},indent=2)+'\n')

if __name__=='__main__':
    main()
