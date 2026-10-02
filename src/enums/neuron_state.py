from enum import Enum


class NeuronState(Enum):
    """
    Represents the high-level state of a neuron based on its action potential interval:
    - RESTING: 0
    - FIRING: 1, 2, 3, 4
    - HYPERPOLARIZATION: 5
    - REFRACTORY: 6, 7, 8, 9, 10
    """

    RESTING = "RESTING"
    FIRING = "FIRING"
    HYPERPOLARIZATION = "HYPERPOLARIZATION"
    REFRACTORY = "REFRACTORY"

    @classmethod
    def from_potential(cls, potential: int) -> "NeuronState":
        if potential == 0:
            return cls.RESTING
        elif 1 <= potential <= 4:
            return cls.FIRING
        elif potential == 5:
            return cls.HYPERPOLARIZATION
        elif 6 <= potential <= 10:
            return cls.REFRACTORY
        else:
            raise ValueError(f"Invalid action potential value: {potential}")

