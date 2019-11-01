import os
import time
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from sklearn.utils import shuffle
import librosa
import numpy as np
import numpy.random as rng


def loadimgs(path, n=0):
    '''
    path => Path of train directory or test directory
    '''
    X = []
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n
    # we load every alphabet seperately so we can isolate them later
    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y, None]
        alphabet_path = os.path.join(path, alphabet)
        # every actor_id/category has it's own column in the array, so  load seperately
        for actor_id in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, actor_id)
            audio_files_counter = 0
            category_images = []
            letter_path = os.path.join(alphabet_path, actor_id)
            # read all the images in the current category
            for actor_dir_name in os.listdir(letter_path):
                filename_path = os.path.join(letter_path, actor_dir_name)
                if audio_files_counter == 10:
                    break

                if not actor_dir_name.startswith('.'):
                    for audio_file in os.listdir(filename_path):
                        image_path = os.path.join(filename_path, audio_file)
                        l, sr = librosa.load(image_path, duration=1)
                        image = np.abs(librosa.stft(l, n_fft=512, hop_length=256))
                        # pad = 259 - image.shape[1]
                        # padded_image = np.pad(image, [(0, 0), (0, pad)], mode='constant')
                        category_images.append(image)
                        audio_files_counter += 1
                        y.append(curr_y)
                        if audio_files_counter == 10:
                            break
            try:
                X.append(np.stack(category_images))
            # edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1

    y = np.vstack(y)
    X = np.stack(X)
    print(1)

    return X, y, lang_dict


Xtrain, y, train_classes = loadimgs(os.getcwd() + '/train_data/')
Xval, yval, val_classes = loadimgs(os.getcwd() + '/val_data/')


def get_batch(batch_size, s="train"):
    """Create batch of n pairs, half same class, half different class"""
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape

    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes, size=(batch_size,), replace=False)

    # initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, w, h, 1)) for i in range(2)]

    # initialize vector for the targets
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = rng.randint(0, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1, n_classes)) % n_classes

        pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)

    return pairs, targets


def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(32, (10, 10), activation='relu', input_shape=input_shape, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (7, 7), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (5, 5), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(2048, activation='sigmoid',
                    kernel_regularizer=l2(1e-3)))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net


model = get_siamese_model((257, 87, 1))
model.summary()

optimizer = Adam(lr=0.00006)
model.compile(loss="binary_crossentropy", optimizer=optimizer)


def make_oneshot_task(N, s="val", language=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape

    indices = rng.randint(0, n_examples, size=(N,))
    if language is not None:  # if language is specified, select characters for that language
        low, high = categories[language]
        if N > high - low:
            raise ValueError("This language ({}) has less than {} letters".format(language, N))
        categories = rng.choice(range(low, high), size=(N,), replace=False)

    else:  # if no language specified just pick a bunch of random letters
        categories = rng.choice(range(n_classes), size=(N,), replace=False)
    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
    test_image = np.asarray([X[true_category, ex1, :, :]] * N).reshape(N, w, h, 1)
    support_set = X[categories, indices, :, :]
    support_set[0, :, :] = X[true_category, ex2]
    support_set = support_set.reshape(N, w, h, 1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image, support_set]

    return pairs, targets


def test_oneshot(model, N, k, s="val", verbose=0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k, N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N, s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct += 1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct, N))
    return percent_correct


# Hyper parameters
evaluate_every = 200  # interval for evaluating on one-shot tasks
batch_size = 32
n_iter = 50  # No. of training iterations
N_way = 3  # how many classes for testing one-shot tasks
n_val = 20  # how many one-shot tasks to validate on
best = -1
model_path = os.getcwd() + '/models/weights/'

print("Starting training process!")
print("-------------------------------------")
t_start = time.time()

for i in range(1, 30):
    (inputs, targets) = get_batch(4)
    loss = model.train_on_batch(inputs, targets)
    if i % 5 == 0:
        print("\n ------------- \n")
        print("Time for {0} iterations: {1} mins".format(i, (time.time() - t_start) / 60.0))
        print("Train Loss: {0}".format(loss))
        val_acc = test_oneshot(model, N_way, n_val, verbose=True)
        model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            best = val_acc
