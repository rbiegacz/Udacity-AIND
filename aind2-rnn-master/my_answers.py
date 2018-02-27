import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras
from math import ceil

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    wwindow_size = window_size - 1
    iter_max = ceil(len(series)/window_size)
#    for iter in range(0, iter_max+1):
#      s_start = iter*wwindow_size
#      s_end = s_start + window_size
#      X.append(series[s_start:s_end])
#      y.append(series[s_end])
#      if s_end < len(series):
#        X.append(series[s_start:s_end])
#        y.append(series[s_end])
#      else:
#        X.append(series[s_start:-1])
#        y.append(series[-1])

    inputs = [ series[i : i + window_size] for i in range(0, len(series) - window_size), windows_size ]
    outputs = [ series[i + window_size] for i in range(0, len(series) - window_size), windows_size ]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
  model = Sequential()
  # 5 LSTM cells
  model.add(LSTM(5, input_shape=(window_size, 1)))
  # 1 result as the output
  model.add(Dense(1))
  return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?', ' ']
    allowed_chars = list(map(chr, range(97, 123))) + punctuation
    text = text.lower()
    text = ''.join(ch for ch in text if ch in allowed_chars)
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [ text[i : i + window_size] for i in range(0, len(text) - window_size, step_size) ]
    outputs = [ text[i + window_size] for i in range(0, len(text) - window_size, step_size) ]

    return inputs,outputs

# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):
  model = Sequential()
  # 5 LSTM cells
  model.add(LSTM(200, input_shape=(window_size, num_chars)))
  # 1 result as the output
  model.add(Dense(num_chars))
  model.add(Activation('softmax'))
  return model
