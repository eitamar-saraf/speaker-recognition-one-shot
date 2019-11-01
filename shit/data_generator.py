from tensorflow.python import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data, labels, batch_size=32, dim=(32, 128, 300), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        y = []
        list_path_temp = []
        for k in indexes:
            list_path_temp.append(self.data[k])
            y.append(self.labels[k])

        # Generate data
        X = self.__data_generation(list_path_temp)
        X = np.array(X)
        y = np.array(y)
        X = np.expand_dims(X, 4)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_lyrics_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        for lyrics in list_lyrics_temp:
            split_lyrics = lyrics.split()
            count = 0
            curr_song = []
            for word in split_lyrics:
                if count == 128:
                    break
                if word in self.word2vec:
                    curr_song.append(self.word2vec.wv.get_vector(word))
                    count += 1
            if count < 128:
                missing = 128 - count
                for miss in range(missing):
                    curr_song.append([0] * 300)
            X.append(curr_song)

        return X
