from pathlib import Path
import numpy as np
from pyedflib import highlevel


class EEGSignalProcessor:
    """Processador de sinais de EEG para leitura, janelamento em epochs e persistência."""

    def __init__(self, fs: float = 512.0):
        self.fs = fs

    @staticmethod
    def load_edf(file_path: str | Path, eeg_only: bool = True):
        """Carrega o arquivo EDF e retorna os sinais como ndarray (canais, amostras),

        junto aos cabeçalhos dos canais e o cabeçalho global.
        Se eeg_only=True, filtra apenas canais com prefixo 'EEG'.
        """
        signals, signal_headers, header = highlevel.read_edf(str(file_path))

        if eeg_only:
            indices = [
                i for i, h in enumerate(signal_headers) if h["label"].startswith("EEG")
            ]
            signals = [signals[i] for i in indices]
            signal_headers = [signal_headers[i] for i in indices]

        return np.array(signals, dtype=np.float32), signal_headers, header

    @staticmethod
    def create_epochs(
        signals: np.ndarray,
        fs: float = 512.0,
        window_sec: float = 2.0,
        overlap: float = 0.0,
    ) -> np.ndarray:
        """Divide a matriz de sinais (canais, amostras) em épocas (trials).

        Retorna:
            np.ndarray de formato (n_epochs, n_canais, n_amostras_por_epoch)
        """
        n_samples_window = int(window_sec * fs)
        step = int(n_samples_window * (1.0 - overlap))
        total_samples = signals.shape[1]

        n_epochs = (total_samples - n_samples_window) // step + 1

        epochs = np.empty(
            (n_epochs, signals.shape[0], n_samples_window), dtype=signals.dtype
        )
        for i in range(n_epochs):
            start = i * step
            epochs[i] = signals[:, start : start + n_samples_window]

        return epochs

    @staticmethod
    def butterworth_bandpass_filter(
        data: np.ndarray,
        lowcut: float = 1.0,
        highcut: float = 50.0,
        fs: float = 512.0,
        order: int = 4,
    ) -> np.ndarray:
        """Aplica um filtro passa-banda Butterworth de ordem especificada (padrão 4ª ordem, 1-50 Hz)

        conforme descrito no artigo base Khaleghi et al. (2021).
        Utiliza scipy.signal se disponível, ou filtragem no domínio da frequência como fallback.
        """
        try:
            from scipy.signal import butter, filtfilt

            b, a = butter(order, [lowcut, highcut], btype="bandpass", fs=fs)
            return filtfilt(b, a, data)
        except ImportError:
            n = len(data)
            freqs = np.fft.rfftfreq(n, d=1.0 / fs)
            fft_vals = np.fft.rfft(data)
            f0 = np.sqrt(lowcut * highcut)
            bw = highcut - lowcut
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = freqs * bw
                ratio = np.where(denom > 0, (freqs**2 - f0**2) / denom, 1e9)
                gain = 1.0 / np.sqrt(1.0 + (ratio) ** (2 * order))
            filtered_fft = fft_vals * gain
            return np.fft.irfft(filtered_fft, n=n)

    @staticmethod
    def save_epochs(
        epochs: np.ndarray,
        output_path: str | Path,
        channel_names: list[str] | None = None,
        fs: float = 512.0,
        window_sec: float = 2.0,
    ):
        """Salva as epochs em formato NumPy (.npy ou .npz comprimido com metadados)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".npz":
            np.savez_compressed(
                output_path,
                epochs=epochs,
                channels=np.array(channel_names) if channel_names else np.array([]),
                fs=fs,
                window_sec=window_sec,
            )
        else:
            np.save(output_path, epochs)
