from enum import Enum


class NeuronState(Enum):
    RESTING = 0
    FIRING = 1
    HYPERPOLARIZATION = 2
    REFRACTORY = 3