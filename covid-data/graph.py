#!/opt/anaconda3/bin/python
import os
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

from spektral.layers import GraphConv
from spektral.utils import localpooling_filter


class GLU(tf.keras.layers.Layer):
    def __init__(self, filters, kernelsize=3):
        super(GLU, self).__init__()
        self.conv_layer = Conv1D(
            filters=filters, kernel_size=kernelsize)

    def call(self, inputs):
        interMatrix0 = self.conv_layer(inputs)
        interMatrix1, interMatrix2 = tf.split(interMatrix0, 2, axis=3)
        gated = (interMatrix1 * tf.nn.sigmoid(interMatrix2))
        return gated


def nstack(tensor, time_train):
    tensor = tf.reshape(
        tensor, [1, (tf.shape(tensor))[0], (tf.shape(tensor))[1]])
    tensor = tf.tile(tensor, [time_train, 1, 1])
    return tensor


def main(time_train, epochs, C_i, C_o, learning_rate, kernel, regenerate=False):
    os.chdir(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

    if (not os.path.exists('/media/data6TB/spandan/data.p')) or regenerate:
        from data_utils import load_data
        X, A, E = load_data(
            DATASET='data-all.json', R=300, SIGMA=1)  # Shapes: (171, 640, 3) (171, 640, 640) (171, 640, 640, 1)
        with open('/media/data6TB/spandan/data.p', 'wb') as pkl:
            pickle.dump((X, A, E), pkl)

    else:
        with open('/media/data6TB/spandan/data.p', 'rb') as pkl:
            X, A, E = pickle.load(pkl)

    districts = A.shape[1]

    # Inputs
    X_in = Input(shape=(districts, C_i), batch_size=time_train)
    E_in = Input(shape=(districts, districts, 1), batch_size=time_train)
    A_in = Input(shape=(districts, districts), batch_size=time_train)

    # Block
    X_i0 = tf.transpose(tf.expand_dims(X_in, axis=0), perm=[0, 2, 1, 3])
    l1 = GLU(filters=2*C_o, kernelsize=kernel)(X_i0)
    X_i1 = tf.squeeze(tf.transpose(l1, perm=[0, 2, 1, 3]))
    E_i1 = E_in[:X_i1.shape[0], :, :, :]
    A_i1 = A_in[:X_i1.shape[0], :, :]
    l1 = GraphConv(channels=C_i, activation='relu')([X_i1, A_i1, E_i1])
    l1 = tf.expand_dims(tf.transpose(l1, perm=[1, 0, 2]), axis=0)
    l1 = GLU(filters=2*C_o, kernelsize=kernel)(l1)

    # Block
    l2 = GLU(filters=2*C_o, kernelsize=kernel)(l1)
    X_i2 = tf.squeeze(tf.transpose(l2, perm=[0, 2, 1, 3]))
    E_i2 = E_in[:X_i2.shape[0], :, :, :]
    A_i2 = A_in[:X_i2.shape[0], :, :]
    l2 = GraphConv(channels=C_i, activation='relu')([X_i2, A_i2, E_i2])
    l2 = tf.expand_dims(tf.transpose(l2, perm=[1, 0, 2]), axis=0)
    l2 = GLU(filters=2*C_o, kernelsize=kernel)(l2)

    # Output layer
    l3 = GLU(filters=2*C_i, kernelsize=(time_train-4*(kernel-1)))(l2)
    X_i3 = tf.squeeze(tf.transpose(l3, perm=[0, 2, 1, 3]))
    final_output = nstack(Dense(C_i)(X_i3), time_train)

    model = Model(
        inputs=[X_in, E_in, A_in], outputs=final_output)
    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  weighted_metrics=['acc'])
    model.summary()

    X_input = X[:time_train, :, :]
    E_input = E[:time_train, :, :, :]
    A_input = localpooling_filter(
        (A[:time_train, :, :]).numpy(), symmetric=True)
    output = nstack(tf.squeeze(X[time_train, :, :]), time_train)

    model.fit([X_input, E_input, A_input],
              output,
              shuffle=False,
              epochs=epochs
              )


if __name__ == '__main__':
    EPOCHS = 10
    LEARNING_RATE = 0.001
    KERNEL_SIZE = 3
    TIME_TRAIN = 150
    C_I = 3
    C_O = 2
    REGEN = False

    main(TIME_TRAIN, EPOCHS, C_I, C_O, LEARNING_RATE, KERNEL_SIZE, REGEN)
