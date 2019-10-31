import os
from sklearn.model_selection import train_test_split
from config import ROOT, join_path


def loadimgs(path, n=0):
    X = []
    y = []
    lang_dict = {}
    curr_y = n
    # we load every alphabet seperately so we can isolate them later
    for celeb in os.listdir(path):
        print("loading celeb: " + celeb)
        lang_dict[celeb] = curr_y
        celeb_path = join_path(path, celeb)
        for interview in os.listdir(celeb_path):
            interview_path = join_path(celeb_path, interview)
            # read all the audio in the current interview
            for filename in os.listdir(interview_path):
                audio_path = join_path(interview_path, filename)
                y.append(curr_y)
                X.append(audio_path)
        curr_y += 1
    return X, y, lang_dict


def get_data():
    data_folder = ROOT + "/data/wav"
    X, y, labels_dict = loadimgs(data_folder)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.50)
    return x_train, x_val, x_test, y_train, y_val, y_test, labels_dict
