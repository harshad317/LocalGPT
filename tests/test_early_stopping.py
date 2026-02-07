from nanochat.early_stopping import EarlyStopping


def test_early_stopping_disables_with_zero_patience():
    es = EarlyStopping(patience=0)
    improved, stop = es.update(1.0)
    assert improved is False
    assert stop is False


def test_early_stopping_triggers_after_patience():
    es = EarlyStopping(patience=2, min_delta=0.0)
    improved, stop = es.update(10.0)
    assert improved is True
    assert stop is False
    improved, stop = es.update(10.0)
    assert improved is False
    assert stop is False
    improved, stop = es.update(10.0)
    assert improved is False
    assert stop is True


def test_early_stopping_resets_on_improvement_with_min_delta():
    es = EarlyStopping(patience=2, min_delta=0.1)
    es.update(1.0)
    es.update(0.95)  # improvement < min_delta, counts as bad eval
    assert es.bad_evals == 1
    improved, stop = es.update(0.89)  # improvement >= min_delta, resets
    assert improved is True
    assert stop is False
    assert es.bad_evals == 0


def test_early_stopping_state_dict_roundtrip():
    es = EarlyStopping(patience=3, min_delta=0.0)
    es.update(1.0)
    es.update(1.0)
    state = es.state_dict()

    es2 = EarlyStopping(patience=3, min_delta=0.0)
    es2.load_state_dict(state)
    assert es2.best == es.best
    assert es2.bad_evals == es.bad_evals

