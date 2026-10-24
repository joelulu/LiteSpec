"""Fresh-process regression for CUDA-triggering training imports in CPU workers."""
import importlib.abc
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import prepare_gdn_subset as subset


def reject_cuda(*args, **kwargs):
    raise AssertionError('CPU subset preparation attempted to initialize/query CUDA')


class RejectTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name+'.') for name in
               ('yunchang', 'litespec.training', 'litespec.distributed', 'litespec.vlm.train_aligned')):
            raise AssertionError('CPU subset preparation imported '+fullname)


# Every fresh spawned process installs these guards BEFORE validation imports.
sys.meta_path.insert(0, RejectTrainingImports())
torch.cuda._lazy_init = reject_cuda
torch.cuda.get_device_capability = reject_cuda
torch.set_num_threads(1)


class Processor:
    def apply_chat_template(self, turns, **kwargs):
        ids = [1,90,91,92,2]
        if not kwargs['add_generation_prompt']:
            ids += list(range(5,14))
        return dict(input_ids=torch.tensor([ids]),pixel_values=torch.ones(1,24),
                    image_grid_thw=torch.tensor([[1,1,1]]))


def init_cpu_validator(*args):
    subset._PROCESSOR = Processor()
    subset._TARGET_CONFIG = SimpleNamespace(image_token_id=91,vision_start_token_id=90,vision_end_token_id=92)
    subset._OPTIONS = dict(max_length=4096,block_size=4)


subset.init_validator = init_cpu_validator


if __name__ == '__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    rows = [dict(id=str(i), image=sys.argv[1],question=f'q{i}',answer='target answer') for i in range(4)]
    results = list(subset.ordered_checks(rows,[3,1,2,0],workers=2,init_args=()))
    assert [i for i,_ in results] == [3,1,2,0]
    assert all(r['reason'] is None and r['answer_tokens']==9 for _,r in results),results
    print('CPU spawned workers passed with CUDA queries and training imports forbidden')
