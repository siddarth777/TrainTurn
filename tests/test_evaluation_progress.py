import pytest
from railway_rl.config import load_config
from railway_rl.evaluation import evaluator


class ProgressRecorder:
    def __init__(self, **kwargs):
        self.total = kwargs['total']
        self.completed = 0
        self.closed = False
        self.tags = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def set_postfix(self, **kwargs):
        self.tags.append(kwargs)

    def update(self, count):
        self.completed += count


@pytest.mark.parametrize('stress,fail', [(False, False), (True, False), (False, True)])
def test_evaluation_progress_counts_completed_episodes_and_closes(tmp_path, monkeypatch, stress, fail):
    bars = []

    def make_progress(**kwargs):
        bar = ProgressRecorder(**kwargs)
        bars.append(bar)
        return bar

    def episode(*args):
        if fail:
            raise RuntimeError('Simulation failed')
        return {'processed_fraction': 1.0, 'weighted_station_delay': 0.0}, [], []

    monkeypatch.setattr(evaluator, 'tqdm', make_progress)
    monkeypatch.setattr(evaluator, 'run_episode', episode)
    config = load_config()
    config['scenario']['trains'] = 1
    if fail:
        with pytest.raises(RuntimeError, match='Simulation failed'):
            evaluator.evaluate(config, 2, 200000, tmp_path, stress=stress)
    else:
        evaluator.evaluate(config, 2, 200000, tmp_path, stress=stress)
    bar = bars[0]
    assert bar.total == 2 * len(evaluator.POLICIES) * (6 if stress else 1)
    assert bar.completed == (0 if fail else bar.total)
    assert bar.closed
    assert {'setting', 'policy', 'scenario'} <= bar.tags[0].keys()
