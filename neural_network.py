import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking, BatchNormalization
from keras.callbacks import TensorBoard
from keras.saving import save_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collect_data import Data

def preprocess(Data: Data):
    # Creates dictionary of actions
    label_map = {label: num for num, label in enumerate(Data.actions)}

    sequences, labels = [], []

    # Loops through each action/word
    for action in Data.actions:

        # In each action loop through the number of videos
        for sequence in range(Data.no_sequences):

            # In each video loop through each frame and add it to the window
            window = []
            for frame_num in range(Data.sequences_length):
                try:
                    res = np.load(os.path.join(Data.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                except FileNotFoundError:
                    # Handle missing frames by skipping
                    continue
                window.append(res)

            # Apply padding or trimming to make all sequences have the same length
            if len(window) < Data.sequences_length:
                padding = [np.zeros_like(window[0])] * (Data.sequences_length - len(window))
                window = window + padding
            else:
                window = window[:Data.sequences_length]

            sequences.append(window)
            labels.append(label_map[action])

    # 90 videos each with 30 frames each with 1662 keypoints from pose face lh and rh
    X = np.array(sequences)

    # Which action is it (hello, thanks, etc.)
    y = to_categorical(labels).astype(int)

    # Split the data into training and testing sets. Test size is 5% is everything
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.05)

    return X_train, X_test, y_train, y_test

'''

Creates model with 3 LSTM layers then 3 fully connected NN layers

'''
def create_model(Data):
    #instantiates sequential api
    model = Sequential()

    #64 lsmt units, true because there is another layer after it
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))

    #64 Fully connected NN neurons
    model.add(Dense(64, activation='relu')) #first layer
    model.add(Dense(32, activation='relu')) #second layer

    '''
    last layer telling us what our model things it is
    will give values that all add up to 1 largest is what 
    NN thinks action is, softmax activation allows for all values 
    to be added to 1 aka show probabilities 
    '''
    model.add(Dense(Data.actions.shape[0], activation='softmax')) 

    return model

'''

Creates model with 3 LSTM layers then 3 fully connected NN layers using Microsoft data set

'''
def create_model_MS(Data):
    #instantiates sequential api
    model = Sequential()

    model.add(Masking(mask_value=0.0, input_shape=(30, 1662)))
    #64 lsmt units, true because there is another layer after it
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(BatchNormalization())
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(BatchNormalization())
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(BatchNormalization())

    #64 Fully connected NN neurons
    model.add(Dense(64, activation='relu')) #first layer
    model.add(Dense(32, activation='relu')) #second layer

    '''
    last layer telling us what our model things it is
    will give values that all add up to 1 largest is what 
    NN thinks action is, softmax activation allows for all values 
    to be added to 1 aka show probabilities 
    '''
    model.add(Dense(Data.actions.shape[0], activation='softmax')) 

    return model



def run_model(model, Data):

    X_train, X_test, y_train, y_test  = preprocess(Data)
    #To be able to see NN accuracy while its training 
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

    #Show what the model looks like
    model.summary()

    #Save the weights 
    save_model(model,'actions.keras')
    return X_test, y_test

