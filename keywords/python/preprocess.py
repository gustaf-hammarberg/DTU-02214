import os
import numpy as np
from scipy.io import wavfile

# Preprocessing parameters
NUM_CLASSES = 3
FRAME_SIZE = 256
FRAME_STRIDE = 256
SPECTROGRAM_HEIGHT = 64
SAMPLE_RATE = 16000
SPECTROGRAM_WIDTH = SAMPLE_RATE // FRAME_STRIDE  # 62
SPECTRUM_MEAN = 6.305
SPECTRUM_STD = 2.493


def preprocess_all(data_dir: str, out_dir: str):
    # Load and preprocess all directories
    other_x, other_y = _preprocess_directory(os.path.join(data_dir, 'other'), class_index=0)
    yes_x, yes_y = _preprocess_directory(os.path.join(data_dir, 'yes'), class_index=1)
    no_x, no_y = _preprocess_directory(os.path.join(data_dir, 'no'), class_index=2)

    # Concatenate and shuffle
    x_all = np.concatenate([other_x, yes_x, no_x])
    y_all = np.concatenate([other_y, yes_y, no_y])
    indices = np.arange(len(x_all))
    np.random.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]

    # Print mean and std of the entire dataset
    print('Mean of the dataset:', np.mean(x_all))
    print('Standard deviation of the dataset:', np.std(x_all))

    # Split into training, validation and test sets (60% train, 20% val, 20% test)
    num_samples = len(x_all)
    num_train = int(0.6 * num_samples)
    num_val = int(0.2 * num_samples)
    x_train = x_all[:num_train]
    y_train = y_all[:num_train]
    x_val = x_all[num_train:num_train + num_val]
    y_val = y_all[num_train:num_train + num_val]
    x_test = x_all[num_train + num_val:]
    y_test = y_all[num_train + num_val:]

    # Save to files
    os.makedirs(out_dir, exist_ok=True)
    np.save(out_dir + 'x_train.npy', x_train)
    np.save(out_dir + 'y_train.npy', y_train)
    np.save(out_dir + 'x_val.npy', x_val)
    np.save(out_dir + 'y_val.npy', y_val)
    np.save(out_dir + 'x_test.npy', x_test)
    np.save(out_dir + 'y_test.npy', y_test)

    # Clean up to reduce memory usage
    del other_x, other_y, yes_x, yes_y, no_x, no_y, x_all, y_all


def _preprocess_directory(data_dir: str, class_index: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess all audio files in a directory for a specific class.
    :param data_dir: Path to the directory containing audio files.
    :param class_index: Class index to be assigned to all files in this directory.
    :return: All spectrograms and their corresponding labels as numpy arrays.
    """
    # Load and preprocess all recordings in the data folder for a specific class
    print('Preprocessing directory: ', data_dir)
    spectrograms = []
    for wav_file in os.listdir(data_dir):
        if wav_file.endswith('.wav'):
            # Load wav file
            sample_rate, sound_data = wavfile.read(os.path.join(data_dir, wav_file))
            if sample_rate != SAMPLE_RATE:
                raise ValueError(f'Expected sample rate of {SAMPLE_RATE}, but got {sample_rate}.')

            # Make it exactly 1 second long
            if len(sound_data) < SAMPLE_RATE:
                padding = np.zeros(SAMPLE_RATE - len(sound_data))
                sound_data = np.concatenate((sound_data, padding))
            else:
                sound_data = sound_data[:SAMPLE_RATE]

            # Preprocess audio
            spectrogram = preprocess_audio(sound_data)

            # Add it to list
            spectrograms.append(spectrogram)

    return np.stack(spectrograms), np.full(len(spectrograms), class_index)


def preprocess_audio(sound_data: np.ndarray) -> np.ndarray:
    """
    Preprocess raw audio data into a spectrogram.
    :param sound_data: Raw audio data as a numpy array.
    :return: Preprocessed spectrogram as a numpy array of shape (SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT).
    """
    # Preprocess data with hamming window and fourier transform
    spectral_frames = []
    for j in range(0, len(sound_data) - FRAME_SIZE, FRAME_STRIDE):
        frame = sound_data[j:j + FRAME_SIZE]
        frame = frame - np.average(frame)
        frame = frame * np.hamming(FRAME_SIZE)
        spectral_frame = np.abs(np.fft.rfft(frame))
        spectral_frame = np.log1p(spectral_frame)
        spectral_frames.append(spectral_frame)

    # Convert to numpy array
    spectrogram = np.array(spectral_frames)
    if spectrogram.shape[0] != SPECTROGRAM_WIDTH:
        raise ValueError(f'Expected spectrogram width of {SPECTROGRAM_WIDTH}, but got {spectrogram.shape[0]}.')

    # Keep the most relevant frequency bins
    spectrogram = spectrogram[:, 1:SPECTROGRAM_HEIGHT + 1]

    # Normalize data
    spectrogram = (spectrogram - SPECTRUM_MEAN) / SPECTRUM_STD

    return spectrogram


if __name__ == '__main__':
    preprocess_all('../data/', 'gen/')
