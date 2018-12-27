import librosa
import numpy as np
import pickle

def melspectrogram(audio):
    spec = librosa.stft(audio, n_fft=512, window='hann', hop_length=256, win_length=512, pad_mode='constant')
    mel_basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80)
    mel_spec = np.dot(mel_basis, np.abs(spec))
    return np.log(mel_spec + 1e-6)

class GZTan:
    def __init__(self, batch_size):
        self.train_batch_num = 0
        self.test_batch_num  = 0
        self.batch_size = batch_size
        # Import data
        print('importing data')
        with open('music_genres_dataset.pkl', 'rb') as f:
            self.train_set = pickle.load(f)
            self.test_set = pickle.load(f)

        self.train_samples = len(self.train_set['labels'])
        self.test_samples = len(self.test_set['labels'])

        print('Number of test samples: {}'.format(self.test_samples))
        print('Number of train samples: {}'.format(self.train_samples))

    def get_test_batch(self):
        start_idx = (self.test_batch_num * self.test_batch_num) % self.test_samples
        end_idx   = ((self.test_batch_num + 1) * self.test_batch_num) % self.test_samples

        self.test_batch_num += 1

        if start_idx >= end_idx:
            samples = self.test_set['data'][start_idx:] + self.test_set['data'][:end_idx]
            spectrograms = [melspectrogram(x) for x in samples]

            return (spectrograms, self.test_set['labels'][start_idx:] + self.test_set['labels'][:end_idx])
        else:
            samples = self.test_set['data'][start_idx:end_idx]
            spectrograms = [melspectrogram(x) for x in samples]

            return (spectrograms, self.train_set['labels'][start_idx:end_idx])

    def get_train_batch(self):
        start_idx = (self.train_batch_num * self.batch_size) % self.train_samples
        end_idx   = ((self.train_batch_num + 1) * self.batch_size) % self.train_samples

        self.train_batch_num += 1

        if start_idx >= end_idx:
            samples = self.train_set['data'][start_idx:] + self.train_set['data'][:end_idx]
            spectrograms = [melspectrogram(x) for x in samples]

            return (spectrograms, self.train_set['labels'][start_idx:] + self.train_set['labels'][:end_idx])
        else:
            samples = self.train_set['data'][start_idx:end_idx]
            spectrograms = [melspectrogram(x) for x in samples]

            return (spectrograms, self.train_set['labels'][start_idx:end_idx])
