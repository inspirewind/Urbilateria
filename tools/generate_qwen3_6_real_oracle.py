#!/usr/bin/env python3
"""Independent CPU Transformers oracle, streaming native BF16 experts and layers.

Uses upstream decoder/attention/cache/norm/router/shared-expert code. Only expert
storage is replaced with selected safetensors slices; no 35B model allocation.
Run with the Transformers environment and compare using tests/qwen3_6_real.rs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import transformers
from safetensors import safe_open
from transformers import Qwen3_5MoeTextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm, Qwen3_5MoeTextRotaryEmbedding,
)


class Checkpoint:
    def __init__(self, directory):
        self.directory = directory
        self.mapping = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
        self.handles = {}

    def slice(self, name):
        shard = self.mapping[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(self.directory / shard, framework='pt', device='cpu')
        return self.handles[shard].get_slice(name)

    def tensor(self, name):
        return self.slice(name)[:]


class StreamedExperts(torch.nn.Module):
    def __init__(self, checkpoint, prefix):
        super().__init__()
        self.checkpoint, self.prefix = checkpoint, prefix

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Same BF16 expert equation and ascending expert accumulation as upstream.
        result = torch.zeros_like(hidden_states)
        for expert in sorted(set(top_k_index.flatten().tolist())):
            tokens, slots = torch.where(top_k_index == expert)
            gate_up = self.checkpoint.slice(self.prefix + '.gate_up_proj')[expert]
            down = self.checkpoint.slice(self.prefix + '.down_proj')[expert]
            gate, up = torch.nn.functional.linear(hidden_states[tokens], gate_up).chunk(2, dim=-1)
            values = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down)
            result.index_add_(0, tokens, (values * top_k_weights[tokens, slots, None]).to(result.dtype))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, nargs='+', default=[9419, 11])
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = json.loads((args.model_dir / 'config.json').read_text())['text_config']
    config = Qwen3_5MoeTextConfig(**cfg)
    config._attn_implementation = 'eager'
    checkpoint = Checkpoint(args.model_dir)
    cache = DynamicCache(config=config)
    rope = Qwen3_5MoeTextRotaryEmbedding(config)
    steps = []
    with torch.no_grad():
        for position, token in enumerate(args.tokens):
            hidden = checkpoint.slice('model.language_model.embed_tokens.weight')[token].reshape(1, 1, -1)
            positions = torch.full((3, 1, 1), position, dtype=torch.long)
            positional = rope(hidden, positions)
            traces = []
            for layer_id in range(config.num_hidden_layers):
                prefix = f'model.language_model.layers.{layer_id}.'
                with torch.device('meta'):
                    layer = Qwen3_5MoeDecoderLayer(config, layer_id)
                state = {name[len(prefix):]: checkpoint.tensor(name) for name in checkpoint.mapping
                         if name.startswith(prefix) and not name.startswith(prefix + 'mlp.experts.')}
                incompatible = layer.load_state_dict(state, strict=False, assign=True)
                assert set(incompatible.missing_keys) == {'mlp.experts.gate_up_proj', 'mlp.experts.down_proj'}
                assert not incompatible.unexpected_keys
                layer.mlp.experts = StreamedExperts(checkpoint, prefix + 'mlp.experts')
                routed = []
                layer.mlp.gate.register_forward_hook(lambda module, inputs, output: routed.extend(output[2].flatten().tolist()))
                hidden = layer(hidden, position_embeddings=positional, position_ids=positions[0],
                               past_key_values=cache, use_cache=True)
                values = hidden.flatten().float()
                traces.append({'first_16': values[:16].tolist(), 'l2_norm': values.norm().item(), 'experts': routed})
                print(f'position={position} layer={layer_id + 1}/40', flush=True)
                del layer, state
            norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            norm.load_state_dict({'weight': checkpoint.tensor('model.language_model.norm.weight')}, assign=True)
            final = norm(hidden).flatten()
            logits = torch.cat([torch.nn.functional.linear(final, checkpoint.slice('lm_head.weight')[i:i+1024])
                                for i in range(0, config.vocab_size, 1024)]).float()
            top_values, top_ids = logits.topk(20)
            samples = sorted(set(range(0, config.vocab_size, 251)) | set(top_ids.tolist()))
            steps.append({'token': token, 'argmax': logits.argmax().item(), 'layers': traces,
                          'top_20': [{'token': i, 'logit': v} for i, v in zip(top_ids.tolist(), top_values.tolist())],
                          'logit_samples': [[i, logits[i].item()] for i in samples]})
    artifact = {'checkpoint': 'Qwen3.6-35B-A3B', 'transformers_version': transformers.__version__,
                'torch_version': torch.__version__, 'generator': Path(__file__).name,
                'upstream_source_sha256': hashlib.sha256(Path(__import__(Qwen3_5MoeDecoderLayer.__module__, fromlist=['']).__file__).read_bytes()).hexdigest(),
                'steps': steps}
    args.output.write_text(json.dumps(artifact, indent=2) + '\n')


if __name__ == '__main__':
    main()
