"""Opt-in pivot experiments. Feedback is auxiliary data, never a Target cache."""
import torch
from torch import nn
from torch.nn import functional as F

MODES = ("A", "B", "C", "D0", "D1", "D2", "E")


class Residual(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.gate = nn.Embedding(1, width)
        nn.init.constant_(self.gate.weight, -4.)

    def forward(self, x):
        return self.up(F.silu(self.down(self.norm(x)))) * self.gate.weight.sigmoid()


class BankRead(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.q = nn.Linear(width, rank, bias=False)
        self.k = nn.Linear(width, rank, bias=False)
        self.v = nn.Linear(width, rank, bias=False)
        self.out = nn.Linear(rank, width, bias=False)

    def forward(self, query, bank, keep=None):
        z = self.norm(bank)
        mask = None if keep is None else keep[:, None, None, :]
        read = F.scaled_dot_product_attention(self.q(query)[:, None],
            self.k(z)[:, None], self.v(z)[:, None], attn_mask=mask)[:, 0]
        return self.out(read)


class LayerRead(nn.Module):
    def __init__(self, width, rank, stages=False):
        super().__init__()
        self.readers = nn.ModuleList([BankRead(width, rank) for _ in range(3 if stages else 1)])
        self.gate = nn.Embedding(1, width)
        nn.init.constant_(self.gate.weight, -4.)
        if stages:
            self.selector = nn.Linear(width, 3)
            nn.init.zeros_(self.selector.weight)
            nn.init.zeros_(self.selector.bias)

    def forward(self, query, banks, masks):
        values = torch.stack([r(query, b, m) for r, b, m in zip(self.readers, banks, masks)], -2)
        if len(self.readers) == 3:
            weights = self.selector(query).float().softmax(-1).to(query.dtype)
            value = (values * weights[..., None]).sum(-2)
        else:
            value = values[..., 0, :]
        return value * self.gate.weight.sigmoid()


class PivotAugmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        spec = config.dlite_config["pivot_augmentation"]
        self.mode = spec["mode"]
        if self.mode not in MODES:
            raise ValueError(f"unknown pivot mode: {self.mode}")
        self.enabled = True
        h, r = config.hidden_size, int(spec.get("rank", 128))
        if r < 1:
            raise ValueError("positive augmentation rank required")
        self.layers = config.dlite_config["target_layer_ids"]
        self.history_positions = int(spec.get("history_positions", 4))
        if self.history_positions < 1:
            raise ValueError("positive history_positions required")
        if self.mode == "C":
            self.delta = Residual(h, r)
        if self.mode in ("D1", "D2"):
            self.feedback_norm = nn.RMSNorm(h)
            self.feedback_down = nn.Linear(h, r, bias=False)
            self.boundary_down = nn.Linear(h, r, bias=False)
            self.feedback_out = nn.Linear(r, h, bias=False)
            self.gate = nn.Embedding(1, h)
            nn.init.constant_(self.gate.weight, -4.)
            self.distance = nn.Embedding(config.block_size + 1, r)
        if self.mode in ("B", "E"):
            self.reads = nn.ModuleList([LayerRead(h, r, self.mode == "E")
                                       for _ in range(config.num_hidden_layers)])
        if self.mode == "B":
            self.time = nn.Embedding(self.history_positions, h)
        if self.mode == "E":
            self.stage_ids = [[i for i, layer in enumerate(self.layers)
                               if (0 if layer < 7 else 1 if layer < 18 else 2) == stage]
                              for stage in range(3)]
            if not all(self.stage_ids):
                raise ValueError("E requires selected layers in all three stages")

    def feedback(self, data):
        raw = data["feedback"][:, 0]  # B,S,H; reject input position, not a true future state
        change = data["correct_embedding"] - data["wrong_embedding"]
        distance = data["distance"].clamp(0, self.distance.num_embeddings - 1)
        latent = self.feedback_down(self.feedback_norm(raw)) + self.boundary_down(change)[:, None]
        latent = latent + self.distance(distance)[:, None]
        value = self.feedback_out(F.silu(latent)) * self.gate.weight.sigmoid()
        return value * data["feedback_valid"][:, None, None].to(value.dtype)

    def inputs(self, pivot, query, data):
        if not self.enabled:
            return pivot, query
        if self.mode == "C":
            if data is None or "previous" not in data:
                raise ValueError("C requires accepted-prefix previous hidden; use the static/video adapter")
            delta = pivot - data["previous"]
            valid = data["previous_valid"][:, None, None, None].to(delta.dtype)
            pivot = pivot + valid * self.delta(delta)
        elif self.mode in ("D1", "D2") and data is not None and "feedback" in data:
            value = self.feedback(data)
            if self.mode == "D1":
                pivot = pivot + value[:, None]
            else:
                # DLite predicts a+1 from query index 1, NOT from the anchor query.
                query = query.clone()
                query[:, 1] = query[:, 1] + value.mean(1)
        return pivot, query

    def layer_residual(self, index, query, raw_pivot, data, depth):
        if not self.enabled or self.mode not in ("B", "E"):
            return query
        if self.mode == "B":
            if data is None or "history" not in data:
                raise ValueError("B requires accepted history; use the static/video adapter")
            bank = data["history"]  # B,K,S,H, excluding the pivot
            time = self.time.weight[None, :, None]
            bank = (bank + depth[None, None] + time).flatten(1, 2)
            masks = [data["history_keep"]]
            banks = [bank]
        else:
            bank = raw_pivot[:, 0] + depth[None]
            banks = [bank[:, ids] for ids in self.stage_ids]
            masks = [None] * 3
        return query + self.reads[index](query, banks, masks)


def gather_positions(source, positions, layers):
    """positions B,N -> B,N,S,H. All gathers are explicit about source index."""
    return torch.stack([source[i].gather(1, positions.clamp(min=0, max=source[i].size(1)-1)
        [..., None].expand(-1, -1, source[i].size(-1))) for i in layers], 2)


def prefix_data(source, anchors, draft):
    aug = draft.pivot_augmentation
    if aug is None:
        return {}
    # Flatten B,N anchors into independent block rows without replicating the
    # full frozen source. The single-anchor inference interface is unchanged.
    b, n = anchors.shape
    result = {}
    if aug.mode == "C":
        result.update(previous=gather_positions(source, anchors-2, draft.target_layer_ids).reshape(b*n,1,draft.chs_num_layers,-1),
                      previous_valid=anchors.flatten() >= 2)
    if aug.mode == "B":
        pos = anchors[..., None] - 1 - aug.history_positions + torch.arange(aug.history_positions, device=anchors.device)
        result.update(history=gather_positions(source, pos.flatten(1), draft.target_layer_ids)
            .reshape(b*n,aug.history_positions,draft.chs_num_layers,-1),
            history_keep=(pos >= 0).reshape(b*n,-1).repeat_interleave(draft.chs_num_layers, -1))
    return result


class AugmentationState:
    """Bounded request-local accepted history and one rejected-position record."""
    def __init__(self, draft, target, states, length):
        from .sw_token import source_layers
        self.draft, self.target = draft, target
        aug = draft.pivot_augmentation
        self.capacity = aug.history_positions+1 if aug.mode == "B" else 2 if aug.mode == "C" else 1
        source = source_layers(states, draft)
        self.source = {i: x[:, -self.capacity:].detach().clone() for i, x in source.items()}
        self.length = length
        self.record = {}

    def data(self):
        # Relative indices preserve the same prefix selection as training.
        n = next(iter(self.source.values())).size(1)
        b = next(iter(self.source.values())).size(0)
        anchors = torch.full((b, 1), n, device=next(iter(self.source.values())).device, dtype=torch.long)
        return dict(prefix_data(self.source, anchors, self.draft), **self.record)

    def capture_feedback(self, states, candidates, kept, predictions):
        from .sw_token import source_layers
        self.record = {}
        if self.draft.pivot_augmentation.mode not in ("D1", "D2") or kept >= candidates.size(1):
            return
        source = source_layers(states, self.draft)
        b = candidates.size(0)
        reject = torch.full((b, 1), kept, device=candidates.device, dtype=torch.long)
        embedding = self.target.get_input_embeddings()
        self.record = dict(feedback=gather_positions(source, reject, self.draft.target_layer_ids).detach().clone(),
            feedback_valid=torch.ones(b, device=candidates.device, dtype=torch.bool),
            wrong_embedding=embedding(candidates[:, kept]).detach().clone(),
            correct_embedding=embedding(predictions[:, kept-1]).detach().clone(),
            distance=torch.ones(b, device=candidates.device, dtype=torch.long))

    def commit(self, states, kept):
        from .sw_token import source_layers
        source = source_layers(states, self.draft)
        self.source = {i: torch.cat([self.source[i], x[:, :kept]], 1)[:, -self.capacity:].detach().clone()
                       for i, x in source.items()}
        self.length += kept
