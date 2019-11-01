import librosa.display
import os

from shit.data_generator import DataGenerator
from shit.data_provider import get_data

x_train, x_val, x_test, y_train, y_val, y_test, labels_dict = get_data()

train_gen = DataGenerator(data=x_train, labels=y_train, n_classes=len(labels_dict.keys()))
val_gen = DataGenerator(data=x_train, labels=y_train, n_classes=len(labels_dict.keys()))
test_gen = DataGenerator(data=x_train, labels=y_train, n_classes=len(labels_dict.keys()))

cwd = os.getcwd()

y, sr = librosa.load(cwd + '/data/01.wav', duration=3.0)
audio = librosa.feature.melspectrogram(y=y, sr=sr)

y_1, sr_1 = librosa.load(cwd + '/data/01.wav', duration=3.0)
audio_1 = librosa.feature.melspectrogram(y=y, sr=sr)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 4))
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()

print('a')
