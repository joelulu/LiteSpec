from litespec.vlm.progress import StepProgress, format_duration


def test_progress_uses_only_steps_in_this_session():
    clock = StepProgress(100, step=80, now=0)
    first = clock.update(82, now=10)
    assert first["seconds_per_optimizer_step"] == 5
    assert first["remaining_steps"] == 18
    assert first["eta_seconds"] == 90
    assert first["eta"] == "00:01:30"
    assert clock.update(100, now=100)["eta_seconds"] == 0
    assert format_duration(None) == "estimating"


def test_watcher_parses_only_optimizer_records():
    from watch_vlm_progress import step_record
    assert step_record('WARNING something') is None
    assert step_record('{"config": {}}') is None
    assert step_record('{"optimizer_step": 92}')["optimizer_step"] == 92
