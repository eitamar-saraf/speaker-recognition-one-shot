import os

import numpy as np
from sklearn.model_selection import train_test_split
from shit.config import ROOT, join_path
import librosa


def loadimgs(path, n=0):
    X = []
    y = []
    lang_dict = {}
    curr_y = n
    # we load every alphabet seperately so we can isolate them later
    for celeb in os.listdir(path):
        counter = 0
        print("loading celeb: " + celeb)
        lang_dict[celeb] = curr_y
        celeb_path = join_path(path, celeb)
        for interview in os.listdir(celeb_path):
            interview_path = join_path(celeb_path, interview)
            # read all the audio in the current interview
            for filename in os.listdir(interview_path):
                audio_path = join_path(interview_path, filename)
                audio, sr = librosa.load(audio_path, duration=3)
                audio = np.abs(librosa.stft(audio, n_fft=512, hop_length=256))
                pad = 259 - audio.shape[1]
                padded_audio = np.pad(audio, [(0, 0), (0, pad)], mode='constant')
                y.append(curr_y)
                X.append(padded_audio)
                counter += 1
                if counter >= 10:
                    break
            if counter >= 10:
                break
        curr_y += 1
    return X, y, lang_dict


def get_data():
    data_folder = ROOT + "/data/wav"
    X, y, labels_dict = loadimgs(data_folder)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.50)
    return x_train, x_val, x_test, y_train, y_val, y_test, labels_dict
