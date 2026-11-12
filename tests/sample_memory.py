"""Per-request diagnostics outside timed model generation; never retains tensors."""
import json
import os
from pathlib import Path
import torch


def snapshot(device):
    if torch.device(device).type != 'cuda':return {}
    return dict(allocated_gib=torch.cuda.memory_allocated(device)/1024**3,
        reserved_gib=torch.cuda.memory_reserved(device)/1024**3,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3)


def measured_request(run, example, cap, *, path, label, index, count, batch_size, pages=None):
    device=example.inputs['input_ids'].device
    # Examples can be CPU-resident and transferred inside run().
    if device.type=='cpu' and torch.cuda.is_available():device=torch.device('cuda',torch.cuda.current_device())
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    info=dict(model=label,index=index,total=count,sample_id=example.sample_id,
        input_tokens=example.prefill_length,visual_tokens=int(example.vision_mask.sum()),
        pages=pages,batch_size=batch_size,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        logical_device=str(device))
    def record(event,**fields):
        row=dict(info,event=event,**fields)
        with Path(path).open('a') as handle:handle.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    record('start',memory=snapshot(device))
    try:
        result=run(example,cap)
    except Exception as exc:
        record('failed',error_type=type(exc).__name__,error=str(exc),memory=snapshot(device))
        raise
    stats=snapshot(device)
    record('complete',memory=stats)
    result['sample_memory']=stats
    return result
