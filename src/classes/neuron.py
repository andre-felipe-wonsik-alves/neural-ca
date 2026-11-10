from enums.neuron_state import NeuronState


class Neuron:
    REST_POTENTIAL = 0
    FIRING_START = 1
    MAX_POTENTIAL = 10

    def __init__(
        self,
        id: int,
        initial_state: int | NeuronState = REST_POTENTIAL,
        is_inhibitory: bool = False,
    ):
        self.id = id
        if isinstance(initial_state, NeuronState):
            state_map = {
                NeuronState.RESTING: 0,
                NeuronState.FIRING: 1,
                NeuronState.HYPERPOLARIZATION: 5,
                NeuronState.REFRACTORY: 6,
            }
            self.potential = state_map[initial_state]
        else:
            self.potential = int(initial_state)
        self.is_inhibitory = is_inhibitory
        self.neighbors: list["Neuron"] = []

    @property
    def state(self) -> NeuronState:
        return NeuronState.from_potential(self.potential)

    @property
    def is_firing(self) -> bool:
        return 1 <= self.potential <= 4

    @property
    def is_hyperpolarized(self) -> bool:
        return self.potential == 5

    def update_state(
        self,
        close_excitatories: int,
        close_inhibitories: int,
        close_hyperpolarizators: int,
        rest_threshold: int = 8,
        relative_threshold: int = 10,
        alpha: float = 0.1,
        spontaneous: bool = False,
    ):
        stimulus = (
            close_excitatories - close_inhibitories - (alpha * close_hyperpolarizators)
        )

        current_potential = self.potential

        if current_potential == self.REST_POTENTIAL:
            if stimulus >= rest_threshold or spontaneous:
                self.potential = self.FIRING_START

        elif 1 <= current_potential <= 5:  # FIRING (1..4) or HYPERPOLARIZATION (5)
            self.potential += 1

        elif 6 <= current_potential <= self.MAX_POTENTIAL:  # REFRACTORY (6..10)
            if stimulus >= relative_threshold:
                self.potential = self.FIRING_START
            else:
                if self.potential >= self.MAX_POTENTIAL:
                    self.potential = self.REST_POTENTIAL
                else:
                    self.potential += 1

    def get_state(self, action_potential: int | None = None) -> NeuronState:
        if action_potential is None:
            return self.state
        return NeuronState.from_potential(action_potential)

