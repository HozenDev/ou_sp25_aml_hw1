from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1, l2

# Define an improved deep_network_basic function
def deep_network_basic(n_inputs:int,
                       n_hidden:list,
                       n_output:int,
                       activation:str='elu',
                       activation_out:str='linear',
                       dropout:float=0.0,
                       dropout_input:float=0.0,
                       kernel_regularizer:float=0.0,
                       kernel_regularizer_L1:float=0.0,
                       metrics=['mse'],
                       lrate:float=0.001) -> Sequential:
    """
    Constructs a sequential neural network model with proper output activation.

    :param n_inputs: Number of input features.
    :param n_hidden: List containing the number of neurons for each hidden layer.
    :param n_output: Number of output neurons.
    :param hidden_activation: Activation function for the hidden layers.
    :param output_activation: Activation function for the output layer.
    :param lrate: Learning rate for the Adam optimizer.
    :return: Compiled Keras sequential model.
    """
    model = Sequential()
    model.add(InputLayer(shape=(n_inputs,)))

    if dropout_input > 0.0:
        model.add(Dropout(dropout_input))
    
    for i, n in enumerate(n_hidden):
        model.add(Dense(n, activation=activation, name=f'hidden{i}'))
        if dropout > 0.0:
            model.add(Dropout(dropout))

    if kernel_regularizer > 0.0:
        model.add(Dense(n_output, activation=activation_out, 
                        name='output', kernel_regularizer=l2(kernel_regularizer)))
    elif kernel_regularizer_L1 > 0.0:
        model.add(Dense(n_output, activation=activation_out, 
                        name='output', kernel_regularizer=l1(kernel_regularizer_L1)))
    else:
        model.add(Dense(n_output, activation=activation_out, name='output'))
    
    opt = Adam(learning_rate=lrate)
    model.compile(loss='mse', optimizer=opt, metrics=metrics)
    
    return model
