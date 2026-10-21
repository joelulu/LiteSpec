"""Read existing single-turn question or turns JSONL for evaluation only.

No dataset is converted or rewritten. Prompt strings are preserved exactly.
Multi-turn records are rejected rather than silently selecting/truncating turns.
"""
from pathlib import Path
from litespec.vlm.runtime import read_records


def read_eval_records(path, limit=None):
    for record in read_records(Path(path).expanduser(), limit):
        if "turns" in record:
            if "question" in record:
                raise ValueError("ambiguous evaluation record: use question or turns, not both")
            turns = record["turns"]
            if not isinstance(turns, list) or len(turns) != 1:
                raise ValueError("this evaluation benchmark requires exactly one user turn per record")
            record = {**record, "question": turns[0]}
        if not isinstance(record.get("question"), str) or not record["question"].strip():
            raise ValueError("evaluation records need a non-empty question or one string in turns")
        yield record
