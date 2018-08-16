import pytest
from pytoune.framework.policies import Phase


def test_phase_raise_without_lr_or_momentum():
    with pytest.raises(ValueError):
        Phase(steps=100, lr=None, momentum=None)

    with pytest.raises(ValueError):
        Phase(steps=100)


@pytest.mark.parametrize("steps,spec,expected", [
    (3, 0, [0, 0, 0]),
    (3, 1, [1, 1, 1]),
    (3, (0, 0), [0, 0, 0]),
    (3, (0, 1), [0, .5, 1]),
    (3, (1, 0), [1, .5, 0]),
    (3, (1, 0, "lin"), [1, .5, 0]),
    (3, (1, 0, "linear"), [1, .5, 0]),
    (3, (1, 0, "cos"), [1, .5, 0]),
    (3, (1, 0, "cosine"), [1, .5, 0]),
])
def test_phase_with_lr_or_momentum(steps, spec, expected):
    phase = Phase(steps, lr=spec)
    assert len(phase) == steps
    assert len(phase.lr) == steps
    for params, exp_value in zip(phase, expected):
        assert len(params.keys()) == 1
        assert "lr" in params
        assert params["lr"] == exp_value

    # same with momentum
    # TODO refactor
    phase = Phase(steps, momentum=spec)
    assert len(phase) == steps
    assert len(phase.momentum) == steps
    for params, exp_value in zip(phase, expected):
        assert len(params.keys()) == 1
        assert "momentum" in params
        assert params["momentum"] == exp_value


def test_phase_with_cosine_lr():
    phase = Phase(5, lr=(1, 0, "cos"))
    assert len(phase) == 5
    assert len(phase.lr) == 5
    assert phase[0]["lr"] == 1
    assert phase[1]["lr"] > .75
    assert phase[2]["lr"] == .5
    assert phase[3]["lr"] < .25
    assert phase[4]["lr"] == 0


# TODO test _parse_spec
# TODO test triangular (or maybe remove it)
# TODO test TrainPolicy
