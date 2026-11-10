from controllers.eeg_signal_processor import EEGSignalProcessor
from controllers.neuron_network import NeuronalNetwork

if __name__ == "__main__":
    # Inicializa a rede 40x40 (1600 neurônios) conforme Khaleghi et al. (2021)
    network = NeuronalNetwork(40, 40)

    # Simulação da dinâmica neuronal (ex.: 1000 iterações conforme Figuras 1 e 2 do artigo)
    steps = 5000
    raw_signal = network.run(steps)

    print(f"--- Sinal Bruto Simulada (Atividade Neuronal da População) ---")
    print(f"Total de iterações: {len(raw_signal)}")
    print(f"Média de neurônios ativos: {raw_signal.mean():.2f}")
    print(f"Desvio padrão: {raw_signal.std():.2f}")
    print(f"Valores iniciais: {raw_signal[:10]}")
    print(f"Valores finais: {raw_signal[-10:]}")

    # Simula o sinal EEG aplicando o filtro passa-banda Butterworth de 4ª ordem (1-50 Hz)
    eeg_simulated = EEGSignalProcessor.butterworth_bandpass_filter(
        raw_signal, lowcut=1.0, highcut=50.0, fs=512.0, order=4
    )

    print(f"\n--- Sinal de EEG Simulado (pós-filtro Butterworth 1-50 Hz) ---")
    print(f"Média: {eeg_simulated.mean():.4f} (centrado em 0)")
    print(f"Amplitude mín / máx: {eeg_simulated.min():.2f} / {eeg_simulated.max():.2f}")
    print(f"Amostras: {eeg_simulated[:10]}")
