from math import cos, pi
import math
import numpy as np
import matplotlib.pyplot as plt

from pytoune.framework.callbacks.callbacks import Callback


###############################################################################
class Phase:
    """
    A Phase just gives you the parameters for an optimizer at step `idx`.

    The interface is similar to the Dataset class.

    TODO generalize to arbitrary parameters not just lr and momentum. Use kwargs?
    """
    def __init__(self, steps, *, lr=None, momentum=None):
        if lr is None and momentum is None:
            raise ValueError("You must specify lr and/or momentum.")

        self.steps = steps
        self.lr, self.lr_spec = self._parse_spec(lr, self.steps)
        self.momentum, self.momentum_spec = self._parse_spec(momentum, self.steps)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        result = {}
        if self.lr_spec is not None:
            result["lr"] = self.lr[idx]
        if self.momentum is not None:
            result["momentum"] = self.momentum[idx]
        return result

    def __repr__(self):
        return (
            f"Phase:\n"
            f"    lr: {self.lr_spec}\n"
            f"    momentum: {self.momentum_spec}"
        )

    def _parse_spec(self, spec, steps):
        if spec is None:
            return None, None

        # clean up spec
        if isinstance(spec, (int, float)):
            spec = (spec, spec, "linear")
        elif isinstance(spec, tuple) and len(spec) == 2:
            spec = (spec[0], spec[1], "linear")

        # init from spec
        if isinstance(spec, tuple) and len(spec) == 3:
            start, end, type_ = spec
            if type_ in ("linear", "lin"):
                return np.linspace(start, end, steps), spec
            elif type_ in ("tri", "triangular"):
                values = np.block([
                    np.linspace(start, end, math.floor(steps / 2)),
                    np.linspace(end, start, math.ceil(steps / 2)),
                ])
                return values, spec
            elif type_ in ("cosine", "cos"):
                # TODO is this correct?
                start, end = end, start
                values = [
                    (start + 0.5 * (end - start) * (1 + cos(i / (steps - 1) * pi)))
                    for i in range(steps)
                ]
                return values, spec

            else:
                raise ValueError(
                    f"The interpolation type most be either one of: "
                    f"'lin', 'cosine', or 'tri'. "
                    f"But it is {type_}"
                )
        else:
            raise ValueError(f"Invalid specification '{spec}'")

    def plot(self, attr="lr", ax=None):
        """
        Plot the phase for the given `attr`.
        """
        if ax is None:
            _fig, ax = plt.subplots()

        ax.plot(list(getattr(self, attr)))
        ax.set_xlabel(attr)
        ax.set_ylabel("steps")
        return ax


###############################################################################
# complex policies build from simple phases
def one_cycle_phases(steps, lr=(0.1, 1, 0.08), momentum=(0.95, 0.85)):
    """
    TODO doc
    """
    steps_up = steps_down = int(steps * .48)
    steps_fine = steps - steps_up - steps_down
    return [
        Phase(steps_up,   lr=(lr[0], lr[1]), momentum=(momentum[0], momentum[1])),
        Phase(steps_down, lr=(lr[1], lr[0]), momentum=(momentum[1], momentum[0])),
        Phase(steps_fine, lr=(lr[0], lr[2]), momentum=momentum[0]),
    ]


def sgdr_phases(base_cycle_length, repeat, lr=(1, 0.1), cycle_mult=2):
    """
    TODO doc

    TODO signature:
    - define the total length and the base_cycle_length instead
    - better names

    total_length = sum(base_cycle_length * (cycle_mult**i)) for i in range(repeat)
    """
    return [
        Phase(base_cycle_length * (cycle_mult**i), lr=(lr[0], lr[1], "cosine"))
        for i in range(repeat)
    ]


###############################################################################
class TrainPolicy(Callback):
    """
    Combine different `Phase`s in a `TrainPolicy`.
    """
    def __init__(self, phases):
        self.phases = phases
        self.phase_iter = self._phase_iter()

    def on_batch_begin(self, batch, logs):
        try:
            spec = next(self.phase_iter)
            self._update_optimizer(spec)
        except StopIteration:
            # Don't do anything when we run out of phases
            # TODO there must be a better way
            pass

    def _phase_iter(self):
        for phase in self.phases:
            for spec in phase:
                yield spec

    def _update_optimizer(self, spec):
        for param_name, param_value in spec.items():
            for group in self.model.optimizer.param_groups:
                group[param_name] = param_value

    def plot(self, attr="lr", ax=None):
        # TODO draw text for each phase
        if ax is None:
            _fig, ax = plt.subplots()

        all_phases = []
        for phase in self.phases:
            all_phases.extend(item[attr] for item in phase)
        ax.plot(all_phases)
        ax.set_ylabel(attr)
        ax.set_xlabel("steps")
