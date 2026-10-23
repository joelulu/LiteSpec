"""Text sampling and serving contracts; no model weights or GPUs required."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_text_control import prepare, sample_rows
from litespec.vlm.serving_data import LengthValidator, chat_payload, read_jsonl, regenerate
from test_vlm_serving import server, Validator


def source_file(tmp_path, count=100):
    path = tmp_path / 'source.jsonl'
    path.write_text(''.join(json.dumps(dict(id=i, conversations=[
        dict(role='user', content=f'question {i}'),
        dict(role='assistant', content='OLD ANSWER DO NOT COPY'),
        dict(role='user', content='LATER QUESTION DO NOT COPY')])) + '\n' for i in range(count)))
    return path


def test_full_source_reservoir_and_first_turn(tmp_path):
    path = source_file(tmp_path)
    rows, scanned = sample_rows(path, 20, 42)
    assert scanned == 100 and len(rows) == 20
    assert rows == sample_rows(path, 20, 42)[0]
    assert rows != sample_rows(path, 20, 43)[0]
    assert max(row['source_line'] for row in rows) > 80
    assert len({row['source_line'] for row in rows}) == 20
    assert 'DO NOT COPY' not in json.dumps(rows)
    with pytest.raises(ValueError, match='only 100'):
        sample_rows(path, 101, 42)


def test_converter_manifest_resume_identity(tmp_path):
    path = source_file(tmp_path)
    directory = tmp_path / 'prepared'
    result = prepare(path, directory, 20, 42)
    records = [row for _, row in read_jsonl(directory / 'prompts.jsonl')]
    assert result['selected'] == len(records) == 20
    assert all(row['split'] == 'train' and 'answer' not in row and 'image' not in row for row in records)
    assert len({row['id'] for row in records}) == 20
    assert result == prepare(path, directory, 20, 42)
    with pytest.raises(ValueError, match='selection changed'):
        prepare(path, directory, 20, 43)


def test_text_payload_and_native_template_contract():
    record = dict(id='text', question='Hello')
    payload = chat_payload(record, 'qwen35-vlm', 256)
    assert payload['messages'] == [dict(role='user', content=[dict(type='text', text='Hello')])]
    assert payload['temperature'] == 0 and payload['chat_template_kwargs']['enable_thinking'] is False
    validator = LengthValidator.__new__(LengthValidator)
    tokens = Mock()
    tokens.__getitem__ = Mock(return_value=SimpleNamespace(tolist=lambda: [1, 2, 3]))
    processor = Mock()
    processor.apply_chat_template.return_value = {'input_ids': tokens}
    validator.processor = processor
    assert validator.encode(record) == [1, 2, 3]
    kwargs = processor.apply_chat_template.call_args.kwargs
    assert kwargs['enable_thinking'] is False and kwargs['add_generation_prompt'] is True
    assert processor.apply_chat_template.call_args.args[0] == payload['messages']


def test_text_rollout_uses_existing_http_and_resume(tmp_path):
    manifest = tmp_path / 'prompts.jsonl'
    manifest.write_text(json.dumps(dict(id='text', question='Hello', split='train')) + '\n')
    with server() as (address, state):
        args = argparse.Namespace(manifest=str(manifest), output_dir=str(tmp_path / 'output'),
            target_model='/local/qwen35', model='qwen35-vlm', server_address=[address],
            concurrency=2, max_tokens=256, max_length=4096, timeout=10,
            retries=0, api_key_env='UNUSED_TEXT_TEST_KEY', image_max_tokens=0)
        assert regenerate(args, Validator())['succeeded'] == 1
        assert regenerate(args, Validator())['already_done'] == 1
        assert len(state['requests']) == 1
    saved = json.loads((tmp_path / 'output/train.jsonl').read_text())
    assert saved['answer'] == 'A picture.' and 'image' not in saved


def test_manifest_pipeline_direct_command_and_cleanup(tmp_path, monkeypatch):
    from litespec.vlm import data_pipeline as pipeline
    manifest, target = tmp_path / 'prompts.jsonl', tmp_path / 'target'
    manifest.write_text('{}\n')
    target.mkdir()
    for key, value in dict(PROMPT_MANIFEST=str(manifest), IMAGE_MAX_TOKENS='0',
        CLIENT_PYTHON=sys.executable, TARGET_MODEL=str(target),
        PREPARE_OUTPUT=str(tmp_path / 'output')).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(pipeline.subprocess, 'run', Mock())
    processes = [Mock(pid=100+i) for i in range(8)]
    for process in processes:
        process.poll.return_value = None
    client = Mock(returncode=0)
    client.poll.return_value = 0
    popen = Mock(side_effect=[*processes, client])
    monkeypatch.setattr(pipeline.subprocess, 'Popen', popen)
    monkeypatch.setattr(pipeline.socket, 'socket', MagicMock())
    monkeypatch.setattr(pipeline.signal, 'signal', Mock())
    monkeypatch.setattr(pipeline, 'wait_ready', Mock())
    cleanup = Mock()
    monkeypatch.setattr(pipeline, 'stop_owned', cleanup)
    pipeline.main()
    command = popen.call_args_list[-1].args[0]
    assert command[0] == sys.executable and command[1].endswith('regenerate_vlm_serving.py')
    assert command[command.index('--manifest')+1] == str(manifest)
    assert len(command[command.index('--server-address')+1:]) == 8
    assert cleanup.call_args.args[0] == processes


def test_training_wrapper_config_and_checkpoint_selectors(tmp_path):
    root = Path(__file__).resolve().parents[1]
    python = tmp_path / 'python'
    calls = tmp_path / 'calls.jsonl'
    python.write_text(f'''#!{sys.executable}
import json, os, sys
with open({str(calls)!r}, 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1:3] == ['-m', 'torch.distributed.run']:
    sys.exit(0)
if len(sys.argv) > 1 and sys.argv[1].endswith('benchmark_vlm_static_batch.py'):
    sys.exit(0)
if sys.argv[1:2] == ['-'] and len(sys.argv) == 3 and sys.argv[2].isdigit():
    sys.exit(0)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
    python.chmod(0o755)
    data, target, output = tmp_path / 'data', tmp_path / 'target', tmp_path / 'run'
    data.mkdir()
    target.mkdir()
    (data / 'train.jsonl').write_text('{}\n')
    env = dict(os.environ, PYTHON_BIN=str(python), MODE='train', PREPARE_OUTPUT=str(data),
               TARGET_MODEL=str(target), OUTPUT_DIR=str(output), CUDA_VISIBLE_DEVICES='0,1,2,3,4,5',
               NPROC_PER_NODE='6', MAX_STEPS='1', MAX_SAMPLES='1', REUSE_VALIDATION_FROM='/stale')
    result = subprocess.run(['bash', str(root / 'tests/run_dflash_text_control.sh')],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    import yaml
    cfg = yaml.safe_load((output / 'config.yaml').read_text())
    assert (cfg['num_epochs'], cfg['batch_size'], cfg['accumulation_steps'], cfg['block_size']) == (3, 2, 4, 8)
    command = next(json.loads(line) for line in calls.read_text().splitlines()
                   if 'torch.distributed.run' in line)
    assert '--nproc-per-node=6' in command
    assert '--max-steps' not in command and '--reuse-validation-from' not in command
    assert command[command.index('--max-samples')+1] == '0'
    calls.unlink()
    env.update(MODELS='ar dflash_text dflash_vlm', DFLASH_TEXT='/text/last',
               DFLASH_VLM='/vision/last', BENCHMARK_MODE='static_batch', BATCH_SIZES='1')
    result = subprocess.run(['bash', str(root / 'tests/compare_vlm_16.sh')],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    command = json.loads(calls.read_text().splitlines()[-1])
    assert command.count('--checkpoint') == 2
    assert command[command.index('dflash_text')+1] == '/text/last'
    assert command[command.index('dflash_vlm')+1] == '/vision/last'
