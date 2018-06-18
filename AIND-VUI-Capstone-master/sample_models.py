from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    # simp_rnn = SimpleRNN(units, activation='relu',
    #    return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # changing from SimpleRNN to GRU to help with vanishing gradients
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)

    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    input = input_data
    # add as many layers as recur_layers defines
    for layer in range(recur_layers):
        rnn_name = 'rnn_%d' % layer
        bn_rnn_name = 'bn_' + rnn_name
        rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name=rnn_name)(input)
        input = BatchNormalization(name=bn_rnn_name)(rnn)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(input)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, name="bidirectional", implementation=2))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim=161, filters=100, kernel_size=11,
                conv_stride=3, conv_border_mode='valid',
                units=100, recur_layers=2, output_dim=29, dropout=0.2):
    """ Build a deep network for speech 
    """

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # Add convolution layers
    conv_1d = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d')(input_data)

    # keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
    # addming Max Pooling layer to make the network smaller
    max_pooling = conv_1d
    #max_pooling = MaxPooling1D(pool_size=5, strides=3)(conv_1d)

    # Add batch normalization to normalize all the inputs
    rnn_data = bn_cnn = BatchNormalization(name='bn_1')(max_pooling)
    # Add Droppout layer - this will help with overfitting
    rnn_data_0 = dropout = Dropout(dropout)(bn_cnn)

    rnn_data_1 = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(rnn_data_0)

    rnn_data = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(rnn_data_1)

    # Add bidirectional recurrent layers
    for layer in range(recur_layers):
        # Create Names
        bi_rnn_name = 'bi_rnn_%d' % (layer + 2)
        bn_rnn_name = 'bn_' + bi_rnn_name
        # create BiDirectional RNN layer
        rnn = Bidirectional(GRU(units, return_sequences=True, name=bi_rnn_name, implementation=2))(rnn_data)
        # Add batch normalization to bidirectional rnn layer
        bnn = BatchNormalization(name=bn_rnn_name)(rnn)
        # Add dropout to avoid overfitting
        rnn_data = Dropout(dropout)(bnn)

    # TODO: Add softmax activation layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn_data)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # TODO: Specify model.output_length
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model