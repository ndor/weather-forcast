import numpy as np
import h5py, sys, os
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential

from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.noise import GaussianDropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization

# from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import Recurrent, LSTM

# TODO: dataset creator, by name of site
def load_weather_data(site_name='SBOK', path_to_hdf5_data_file='sites_weather.hdf5'):
    '''
    :param site_name:
        # ARAD
        # BESR
        # BRSH
        # ELAT
        # HZVA
        # MIZP
        # SBOK
        # SDOM
        # YOTV
    :param path_to_hdf5_data_file:
    :return: training_set, testing_set
    '''

    # current_directory_name = os.getcwd()
    site_weather = np.array(h5py.File(path_to_hdf5_data_file)[site_name], dtype='int')
    print site_weather.shape
    site_weather[:, :6][site_weather[:, :6] < 0] = 0 # DNI / IDNI / wind / humidity,  cannot be < 0
    for i in range(len(site_weather)):
        site_weather[i, :][site_weather[i, :] == -999] = site_weather[i - 1, :][site_weather[i, :] == -999]
        if site_weather[i, 6] < 0:
            site_weather[i, 6] = site_weather[i - 1, 6]
        if site_weather[i, 7] < 0:
            site_weather[i, 7] = site_weather[i - 1, 7]
        if site_weather[i, 8] < 0:
            site_weather[i, 8] = site_weather[i - 1, 8]
        if site_weather[i, 9] == -9 and abs(site_weather[i, 9] - site_weather[i - 1, 9]) > 5: # sensor failure
            site_weather[i, 9] = site_weather[i - 1, 9]

    # vector of scalars:
    site_weather[:, -2] = site_weather[:, -2] + 273 # for normalization of 0.0 to 1.0 we change temperature from Celcius to Kelvins
    multiplication_scalars = [1./10000, 1./12, 1./31, 1./24, 1./10000, 1./10000, 1./10000, 1./10000, 1./1000, 1./100]
    # now normalizing:
    site_weather = site_weather *  multiplication_scalars
    # print np.amax(site_weather, 0)

    training_set = site_weather[:-24*365+12]
    testing_set = site_weather[-24*365+12:]

    return training_set, testing_set


# def train_test_split(df, test_size=0.1):
#     """
#     This just splits data to training and testing parts
#     """
#     ntrn = round(len(df) * (1 - test_size))
#
#     X_train, y_train = _load_data(df.iloc[0:ntrn])
#     X_test, y_test = _load_data(df.iloc[ntrn:])
#
#     return (X_train, y_train), (X_test, y_test)


def X_Y_devision(data, n_prev):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data[i:i+n_prev])
        docY.append(data[i+n_prev])
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


# TODO: net trainer (LSTM NN)
def weather_trainer():

    train, test = load_weather_data()
    number_of_steps_back = 24 * 30 # needed historical number of hours
    nb_IO_neurons = train.shape[1]
    hidden_neurons = 32

    X_train, y_train = X_Y_devision(train, number_of_steps_back)

    print 'number of samples:'
    print len(X_train)
    print '\n sample shape:'
    print X_train[0].shape

    model = Sequential()
    model.add(LSTM(hidden_neurons, return_sequences=True, input_shape=X_train[0].shape))
    model.add(LSTM(hidden_neurons, return_sequences=True))
    model.add(LSTM(hidden_neurons, return_sequences=True))
    model.add(LSTM(hidden_neurons, return_sequences=True))
    model.add(LSTM(hidden_neurons, return_sequences=True))
    model.add(LSTM(hidden_neurons, return_sequences=False))
    # TODO: TimeDistributedDense / Highway
    # TODO: in LSTM properties: LSTM(... , stateful=True, ...) :A stateful recurrent model is one for which the internal states (memories) obtained after processing a batch of samples are reused as initial states for the samples of the next batch. This allows to process longer sequences while keeping computational complexity manageable.
    # model.add(Dense(hidden_neurons, activation='hard_sigmoid'))
    # model.add(Dense(hidden_neurons, activation='hard_sigmoid'))
    model.add(Dense(nb_IO_neurons, activation='linear'))

    optimizer_method = SGD(lr=5e-6, decay=1e-6, momentum=0.9,
                           nesterov=True)  # Adagrad()#Adadelta()#RMSprop()#Adam()#Adadelta()#
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])#optimizer_method)

    ############################################################################################
    # # if previus file exist:
    # if os.path.isfile('generic_run.hdf5'):
    #     print '\n loading weights file: generic_run.hdf5'
    #     model.load_weights('generic_run.hdf5')
    ############################################################################################


    # and now train the model
    # batch_size should be appropriate to your memory size
    # number of epochs should be higher for real world problems
    EarlyStopping(monitor='val_loss', patience=0, verbose=1)
    checkpointer = ModelCheckpoint('generic_run.hdf5', verbose=1, save_best_only=True)
    model.fit(X_train, y_train, batch_size=500, nb_epoch=10000,
              validation_split=0.05, verbose=1, callbacks=[checkpointer])


# load_weather_data()
weather_trainer()

# TODO: net tester
