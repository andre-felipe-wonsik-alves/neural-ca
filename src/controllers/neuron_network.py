import numpy as np

from classes.neuron import Neuron


class NeuronalNetwork:
    """Cellular Automaton neuronal network model for brainwave simulation

    based on Khaleghi et al. (2021).
    """

    def __init__(
        self,
        rows: int = 40,
        columns: int = 40,
        n_min: int = 14,
        n_max: int = 60,
        rest_threshold: int = 8,
        relative_threshold: int = 10,
        alpha: float = 0.1,
        spontaneous_rate: float = 0.0,
    ):
        self.rows: int = rows
        self.columns: int = columns
        self.number_neurons: int = rows * columns
        self.n_min: int = n_min
        self.n_max: int = n_max
        self.rest_threshold: int = rest_threshold
        self.relative_threshold: int = relative_threshold
        self.alpha: float = alpha
        self.spontaneous_rate: float = spontaneous_rate

        self.neurons: list[Neuron] = []
        self._adj_matrix: np.ndarray | None = None

        self._initialize_population()
        self._initialize_connections()

    def _initialize_population(self):
        """Initializes 80% excitatory and 20% inhibitory neurons with action potential

        states distributed according to optimal proportions in Khaleghi et al.:
        - 20% Resting (state 0)
        - 40% Firing (states 1, 2, 3, 4: 10% each)
        - 5% Hyperpolarization (state 5)
        - 35% Refractory (states 6, 7, 8, 9, 10: 7% each)
        """
        # 80% excitatory (False) and 20% inhibitory (True)
        types = np.random.choice(
            [False, True], size=self.number_neurons, p=[0.8, 0.2]
        )

        state_probabilities = [0.20] + [0.10] * 4 + [0.05] + [0.07] * 5
        initial_states = np.random.choice(
            np.arange(11), size=self.number_neurons, p=state_probabilities
        )

        self.neurons = [
            Neuron(
                id=i,
                initial_state=int(initial_states[i]),
                is_inhibitory=bool(types[i]),
            )
            for i in range(self.number_neurons)
        ]

    def _initialize_connections(self):
        """Builds directed synaptic connections for each neuron.

        Each neuron receives k inputs uniformly sampled between n_min and n_max.
        """
        adj = np.zeros((self.number_neurons, self.number_neurons), dtype=np.float32)

        for i, neuron in enumerate(self.neurons):
            k = np.random.randint(self.n_min, self.n_max + 1)
            # Candidates for incoming neighbors (excluding the neuron itself)
            candidates = np.delete(np.arange(self.number_neurons), i)
            chosen_indices = np.random.choice(candidates, size=k, replace=False)

            neuron.neighbors = [self.neurons[idx] for idx in chosen_indices]
            adj[i, chosen_indices] = 1.0

        self._adj_matrix = adj

    def step(self):
        """Executes one discrete synchronous Cellular Automaton time step."""
        inputs = []
        spontaneous_flags = (
            np.random.rand(self.number_neurons) < self.spontaneous_rate
            if self.spontaneous_rate > 0
            else [False] * self.number_neurons
        )

        for neuron in self.neurons:
            close_excitatories = sum(
                1
                for neighbor in neuron.neighbors
                if not neighbor.is_inhibitory and neighbor.is_firing
            )
            close_inhibitories = sum(
                1
                for neighbor in neuron.neighbors
                if neighbor.is_inhibitory and neighbor.is_firing
            )
            close_hyperpolarizators = sum(
                1 for neighbor in neuron.neighbors if neighbor.is_hyperpolarized
            )

            inputs.append(
                (close_excitatories, close_inhibitories, close_hyperpolarizators)
            )

        for neuron, (ce, ci, ch), spont in zip(
            self.neurons, inputs, spontaneous_flags
        ):
            neuron.update_state(
                close_excitatories=ce,
                close_inhibitories=ci,
                close_hyperpolarizators=ch,
                rest_threshold=self.rest_threshold,
                relative_threshold=self.relative_threshold,
                alpha=self.alpha,
                spontaneous=spont,
            )

    def run(
        self,
        steps: int,
        signal_type: str = "active_neurons",
        fast: bool = True,
    ) -> np.ndarray:
        """Runs the simulation for a number of steps and extracts the time series.

        Args:
            steps: Number of iterations to simulate.
            signal_type: 'active_neurons' (number of firing neurons, matching paper
                Figures 1 & 2) or 'potential_sum' (sum of all neuron potentials).
            fast: If True, uses vectorized NumPy matrix multiplications for high
                speed.

        Returns:
            np.ndarray: Time series signal of length `steps`.
        """
        if fast and self._adj_matrix is not None:
            return self._run_vectorized(steps, signal_type)

        time_series = []
        for _ in range(steps):
            self.step()

            if signal_type == "active_neurons":
                val = sum(1 for n in self.neurons if n.is_firing)
            else:
                val = sum(n.potential for n in self.neurons)

            time_series.append(val)

        return np.array(time_series, dtype=np.float32)

    def _run_vectorized(
        self, steps: int, signal_type: str = "active_neurons"
    ) -> np.ndarray:
        """Fast vectorized simulation using NumPy matrix operations."""
        states = np.array([n.potential for n in self.neurons], dtype=np.int32)
        is_inh = np.array([n.is_inhibitory for n in self.neurons], dtype=bool)
        is_exc = (~is_inh).astype(np.float32)
        is_inh_f = is_inh.astype(np.float32)

        A = self._adj_matrix
        T_rest = self.rest_threshold
        T_relative = self.relative_threshold
        alpha = self.alpha
        spont_rate = self.spontaneous_rate
        n_neurons = self.number_neurons

        time_series = np.empty(steps, dtype=np.float32)

        for t in range(steps):
            firing = ((states >= 1) & (states <= 4)).astype(np.float32)

            if signal_type == "active_neurons":
                time_series[t] = float(np.sum(firing))
            else:
                time_series[t] = float(np.sum(states))

            hyper = (states == 5).astype(np.float32)

            stimulus = (
                A @ (firing * is_exc)
                - A @ (firing * is_inh_f)
                - alpha * (A @ hyper)
            )

            next_states = states.copy()

            # Resting neurons (state 0)
            rest_mask = states == 0
            rest_activate = rest_mask & (stimulus >= T_rest)
            if spont_rate > 0:
                spont = rest_mask & (np.random.rand(n_neurons) < spont_rate)
                rest_activate = rest_activate | spont
            next_states[rest_activate] = 1

            # Firing (1..4) and Hyperpolarization (5) -> advance sequentially
            in_cycle = (states >= 1) & (states <= 5)
            next_states[in_cycle] = states[in_cycle] + 1

            # Refractory neurons (6..10)
            refr_mask = (states >= 6) & (states <= 10)
            refr_activate = refr_mask & (stimulus >= T_relative)
            next_states[refr_activate] = 1

            refr_advance = (
                refr_mask & (~refr_activate) & (states < Neuron.MAX_POTENTIAL)
            )
            next_states[refr_advance] = states[refr_advance] + 1

            refr_reset = (
                refr_mask & (~refr_activate) & (states == Neuron.MAX_POTENTIAL)
            )
            next_states[refr_reset] = 0

            states = next_states

        # Synchronize states back to Neuron objects
        for neuron, st in zip(self.neurons, states):
            neuron.potential = int(st)

        return time_series
