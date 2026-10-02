import numpy

from classes.neuron import Neuron


class NeuronalNetwork:
    def __init__(self, rows: int = 40, columns: int = 40):
        self.number_neurons: int = rows * columns
        self.neurons: list[Neuron] = []

    def _initialize_population(self):
        # 80% excitatories and 20% inhibitories
        types = numpy.random.choice(
            [False, True], size=self.number_neurons, p=[0.8, 0.2]
        )

        # ?? ver depois de faz sentido
        # ?? tentei definir o que está no artigo
        states_pool = (
            [0] * 200 + [18, 19, 20, 21] * 100 + [22] * 50 + [2, 4, 5, 23, 24] * 70
        )

        initial_states = numpy.random.choice(states_pool, size=self.number_neurons)

        for i in range(self.number_neurons):
            self.neurons.append(Neuron(i, initial_states[i], types[i]))

    def step(self):
        inputs = []

        for neuron in self.neurons:
            close_excitatories = sum(1 for neighbor in neuron.neighbors if not neighbor.is_inhibitory and 1 <= neighbor.state <= 4)
            close_inhibitories = sum(1 for neighbor in neuron.neighbors if neighbor.is_inhibitory and 1 <= neighbor.state <= 4)
            close_hyperpolarizators = sum(1 for neighbor in neuron.neighbors if neighbor.state == 5)
            
            inputs.append((close_excitatories, close_inhibitories, close_hyperpolarizators))


        for neuron, (
            close_excitatories,
            close_inhibitories,
            close_hyperpolarizators,
        ) in zip(self.neurons, inputs):
            neuron.update_state(
                close_excitatories, close_inhibitories, close_hyperpolarizators
            )

    def run(self, steps: int) -> numpy.ndarray:
        time_series = []

        for _ in range(steps):
            self.step()

            # ?? verificar essa soma depois. Como meu state é um enum, não posso fazer um somatório diretamente nele.
            potential_sum = sum(n.potential for n in self.neurons)

            time_series.append(potential_sum)

        return numpy.array(time_series)
