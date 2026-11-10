"""Bounded recent Target positions, not Draft guesses or a persistent Draft KV cache."""
from pathlib import Path
import hashlib
import json

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from litespec.core.dlite import OnlineDLiteModel, _hidden_at_layer


def set_window(draft, size):
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError('context window must be a positive integer')
    if not draft.is_student or getattr(draft, 'visual_kv', None) is not None or getattr(draft, 'memory_reader', None) is not None:
        raise ValueError('window experiment requires an original DLite student')
    draft.context_window_size = size
    draft.config.dlite_config['context_window_size'] = size
    return draft


def source_layers(states, draft):
    return {i: _hidden_at_layer(states, i, draft.config.num_target_layers) for i in draft.target_layer_ids}


def gather_window(source, anchors, layers, size):
    """B,N,K*S,H; oldest time first, increasing layer order within each time."""
    positions = anchors[..., None] - size + torch.arange(size, device=anchors.device)
    valid = positions >= 0
    pieces = []
    for i in layers:
        raw = source[i]
        idx = positions.clamp(min=0, max=raw.size(1)-1)
        pieces.append(raw.gather(1, idx.flatten(1)[..., None].expand(-1, -1, raw.size(-1)))
                      .reshape(*positions.shape, raw.size(-1)))
    features = torch.stack(pieces, -2).flatten(2, 3)
    keep = valid.repeat_interleave(len(layers), -1)
    # Padded anchors may be zero; keep a dummy slot to avoid fully masked rows.
    keep[..., -1] |= ~keep.any(-1)
    return features, keep


class RecentTargetWindow:
    """Small per-request raw window. Prefill and commits slice BEFORE layer stacking."""
    def __init__(self, draft, states, prefix_length):
        self.draft = draft
        self.size = draft.context_window_size
        self.committed = int(prefix_length)
        if self.committed < 1:
            raise ValueError('a processed prefix/pivot is required')
        source = source_layers(states, draft)
        if any(x.size(1) != self.committed for x in source.values()):
            raise ValueError('prefill hidden length differs from prefix')
        self.values = torch.stack([source[i][:, -self.size:] for i in draft.target_layer_ids], 2).detach().clone()

    def context(self):
        b,t,s,h = self.values.shape
        padded = F.pad(self.values, (0,0,0,0,self.size-t,0))
        keep = torch.arange(self.size,device=padded.device) >= self.size-t
        return padded.reshape(b,1,self.size*s,h), keep.repeat_interleave(s)[None].expand(b,-1)

    def commit(self, states, kept, anchor):
        if int(anchor) != self.committed or int(kept) < 1:
            raise ValueError('commit must continue the confirmed prefix exactly once')
        source = source_layers(states,self.draft)
        if any(x.size(1)<kept for x in source.values()):
            raise ValueError('commit exceeds verified hidden length')
        new = torch.stack([source[i][:, max(0,kept-self.size):kept] for i in self.draft.target_layer_ids],2).detach()
        self.values = torch.cat([self.values,new],1)[:, -self.size:].clone()
        self.committed += int(kept)


class OnlineSWToken(OnlineDLiteModel):
    """Same objective/head, independent anchor batches with checkpointed SDPA chunks.

    Frozen selected-layer source is shared. K*S activations are gathered for one
    chunk at a time, not replicated up front for every sampled anchor.
    """
    def __init__(self,*args,**kwargs):
        kwargs['flex_attention_backend']='TRITON'
        super().__init__(*args,**kwargs)
        self.chunk_blocks = int(self.draft_model.config.dlite_config.get('sw_train_chunk_blocks',16))
        if self.chunk_blocks < 1:
            raise ValueError('positive sw_train_chunk_blocks required')

    def prepare_batch(self, input_ids, hidden_states, loss_mask, **kwargs):
        if hidden_states is None:
            raise ValueError('windowed training requires frozen Target hidden states')
        batch = super().prepare_batch(input_ids, hidden_states, loss_mask, **kwargs)
        batch.sw_source = {i: h.detach() for i,h in source_layers(hidden_states,self.draft_model).items()}
        return batch

    def forward_backbone(self,batch,*,seq_len):
        draft=self.draft_model
        if batch.sw_source is None:
            raise ValueError('windowed training requires prepared sw_source; use OnlineSWToken.prepare_batch')
        b,n=batch.anchor_positions.shape
        chunks=[]
        for start in range(0,n,self.chunk_blocks):
            end=min(n,start+self.chunk_blocks)
            count=end-start
            anchors=batch.anchor_positions[:,start:end]
            def run(query, anchors, token_positions, token_keep, chunk_count=count):
                context,keep=gather_window(batch.sw_source,anchors,draft.target_layer_ids,draft.context_window_size)
                cp,qp=draft.build_block_position_ids(anchors,token_positions,token_keep)
                query=query.reshape(b*chunk_count,draft.draft_query_length,-1)
                context=context.reshape(b*chunk_count,1,draft.condition_slot_count,-1)
                allowed=torch.cat([keep.reshape(b*chunk_count,-1),torch.ones(b*chunk_count,draft.draft_query_length,device=query.device,dtype=torch.bool)],-1)
                mask=query.new_zeros(b*chunk_count,1,draft.draft_query_length,allowed.size(-1))
                mask.masked_fill_(~allowed[:,None,None],torch.finfo(query.dtype).min)
                cp=cp.reshape(b*chunk_count,-1);qp=qp.reshape(b*chunk_count,-1)
                output=draft(noise_embedding=query,target_hidden=context,
                    history_hidden=query.new_empty(b*chunk_count,1,0,query.size(-1)),
                    position_ids=qp,rotary_position_ids=torch.cat([cp,qp],-1),attention_mask=mask,is_causal=False)
                return draft._prediction_hidden(output).reshape(b,chunk_count,draft.proposal_length,-1)
            fn = run
            args=(batch.query_embeddings[:,start:end],anchors,batch.token_position_ids[:,start:end],batch.token_keep_mask[:,start:end])
            if self.training and torch.is_grad_enabled():
                chunks.append(checkpoint(fn,*args,use_reentrant=False))
            else:
                chunks.append(fn(*args))
        return torch.cat(chunks,1)



def warm_start_config(cfg,path):
    if cfg.get('algorithm') == 'dflash':
        from .dflash_warmstart import warm_start_config as configure_dflash
        return configure_dflash(cfg, path)
    if cfg.get('algorithm','dlite')!='dlite':
        raise ValueError('warm start currently supports DLite only')
    path=Path(path)
    parent=json.loads((path/'config.json').read_text())
    d=parent.get('dlite_config',{})
    if parent.get('architectures')!=['DLiteDraftModel'] or d.get('model_role')!='student' or d.get('experiment_name') or d.get('memory_mode','none')!='none':
        raise ValueError('warm start requires an original DLite student checkpoint')
    cfg.update(init_checkpoint=str(path.resolve()),init_config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
        block_size=parent['block_size'],num_draft_layers=parent['num_hidden_layers'],
        chs_num_layers=d['chs_num_layers'],target_layer_ids=d['target_layer_ids'],
        sequential_rank=d['sequential_rank'],architecture_version=d.get('architecture_version','dlite_v1'))
    return parent


def load_warm_start(path,target,cfg):
    if cfg.get('algorithm') == 'dflash':
        from .dflash_warmstart import load_warm_start as load_dflash
        return load_dflash(path, target, cfg)
    from transformers import AutoConfig
    from litespec.modeling.draft.dlite import DLiteDraftModel
    config=AutoConfig.from_pretrained(path)
    text=target.config.text_config
    if (config.hidden_size,config.vocab_size,config.num_target_layers)!=(text.hidden_size,text.vocab_size,text.num_hidden_layers):
        raise ValueError('warm-start checkpoint and Target dimensions differ')
    recorded=getattr(config,'vlm_config',{}).get('target_model')
    if recorded and Path(recorded).expanduser().resolve()!=Path(cfg['target_model']).expanduser().resolve():
        raise ValueError('warm-start checkpoint Target path differs')
    config.dlite_config['context_window_size']=int(cfg.get('context_window_size',1))
    config.dlite_config['sw_train_chunk_blocks']=int(cfg.get('sw_train_chunk_blocks',16))
    draft,info=DLiteDraftModel.from_pretrained(path,config=config,attn_implementation='sdpa',output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys')):
        raise ValueError(f'warm start must load every parameter exactly: {info}')
    draft.config.vlm_config=dict(getattr(config,'vlm_config',{}),target_model=cfg['target_model'],
        warm_start=str(Path(path).resolve()),context_window_size=draft.context_window_size)
    return draft.to(device=target.device,dtype=target.dtype)


def configure_image_tokens(processor,config,tokens):
    if tokens < 1:
        raise ValueError('positive image budget required')
    image=processor.image_processor
    size=dict(image.size)
    if not {'shortest_edge','longest_edge'} <= size.keys():
        raise ValueError('unsupported Qwen image processor size policy')
    unit=config.vision_config.patch_size * config.vision_config.spatial_merge_size
    size['longest_edge']=tokens*unit*unit
    size['shortest_edge']=min(size['shortest_edge'],size['longest_edge'])
    image.size=size
    return size

