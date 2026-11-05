"""A/B/C/E: one frozen Target pass, shared hidden, original SFT objective."""
import torch
from torch.utils.checkpoint import checkpoint

from .sw_token import OnlineSWToken
from .pivot_augmentation import prefix_data


class OnlinePivotSFT(OnlineSWToken):
    def forward_backbone(self, batch, *, seq_len):
        d = self.draft_model
        if batch.sw_source is None:
            raise ValueError("shared SFT requires declared frozen sw_source")
        b, n = batch.anchor_positions.shape
        chunks = []
        for start in range(0, n, self.chunk_blocks):
            end = min(start+self.chunk_blocks, n)
            count = end-start

            def run(query, pivot, anchors, positions, keep, chunk_count=count):
                data = prefix_data(batch.sw_source, anchors, d)
                cp, qp = d.build_block_position_ids(anchors, positions, keep)
                query = query.reshape(b*chunk_count, d.draft_query_length, -1)
                pivot = pivot.reshape(b*chunk_count, 1, d.chs_num_layers, -1)
                out = d(noise_embedding=query, target_hidden=pivot,
                    history_hidden=query.new_empty(b*chunk_count, 1, 0, query.size(-1)),
                    position_ids=qp.reshape(b*chunk_count,-1),
                    rotary_position_ids=torch.cat([cp.reshape(b*chunk_count,-1),qp.reshape(b*chunk_count,-1)],-1),
                    attention_mask=None, is_causal=False, augmentation_data=data)
                return d._prediction_hidden(out).reshape(b,chunk_count,d.proposal_length,-1)

            args = (batch.query_embeddings[:,start:end], batch.target_hidden[:,start:end],
                    batch.anchor_positions[:,start:end], batch.token_position_ids[:,start:end],
                    batch.token_keep_mask[:,start:end])
            chunks.append(checkpoint(run,*args,use_reentrant=False)
                          if self.training and torch.is_grad_enabled() else run(*args))
        return torch.cat(chunks,1)
