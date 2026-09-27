from enums.neuron_state import NeuronState


class Neuron:
    def __init__(self, id: int, initial_state: NeuronState, is_inhibitory: bool):
        self.id = id
        self.state = initial_state
        self.is_inhibitory = is_inhibitory
        self.neighbors = []


    def update_state(self, close_excitatories: int, close_inhibitories: int, close_hyperpolarizators:int, rest_threshold: int = 8, relative_threshold: int = 12, alpha: float = 0.1):
        stimulus = close_excitatories - close_inhibitories - (alpha * close_hyperpolarizators)

        if self.state == NeuronState.RESTING:
            if stimulus >= rest_threshold:
                self.state = NeuronState.FIRING
        elif self.state == NeuronState.FIRING or self.state == NeuronState.HYPERPOLARIZATION:
            pass
            # aumentar em um o valor (nao sei como)
        else:
            if stimulus >= relative_threshold:
                self.state = NeuronState.FIRING
            else:
                self.state = NeuronState.RESTING if self.state == NeuronState.REFRACTORY else self.state + 1


    def get_state(self, action_potential: int) -> NeuronState:
        if action_potential == 0:
            return NeuronState.RESTING
        elif 1 <= action_potential <= 4:
            return NeuronState.FIRING
        elif action_potential == 5:
            return NeuronState.HYPERPOLARIZATION
        elif 6 <= action_potential <= 10:
            return NeuronState.REFRACTORY
        else:
            raise ValueError(f"Invalid value: {action_potential}")