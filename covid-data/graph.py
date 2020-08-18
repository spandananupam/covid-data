#!/opt/anaconda3/bin/python
from data_utils import load_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from spektral.layers import GraphConv


nodeFeatures_train, nodeFeatures_test, weightedAdjacency = load_data(
    DATASET='data-all.json', R=300, SIGMA=1, TEST_NUMBER=150)

districts = int(weightedAdjacency.shape[0])
time_train = int(nodeFeatures_train.shape[0])
epochs = 10


class GLU(keras.layers.Layer):
    def __init__(self, filters=160, kernelsize=3):
        super(GLU, self).__init__()
        self.conv_layer = Conv1D(
            filters=filters, kernel_size=kernelsize)

    def call(self, inputs):
        interMatrix1 = self.conv_layer(inputs)
        interMatrix2 = self.conv_layer(inputs)
        gated = (interMatrix1 * tf.nn.sigmoid(interMatrix2))
        return gated


# Inputs
X_in = Input(shape=(time_train, districts))
W_in = Input(shape=(districts, districts), sparse=True)

# Block
l1 = GLU(filters=districts, kernelsize=3)(X_in)
l1 = GraphConv(channels=int(districts))([l1, W_in])
l1 = GLU(filters=districts, kernelsize=3)(l1)

# Block
l2 = GLU(filters=districts, kernelsize=3)(l1)
l2 = GraphConv(channels=int(districts))([l2, W_in])
l2 = GLU(filters=districts, kernelsize=3)(l2)


model = Model(
    inputs=[X_in, W_in], outputs=l2)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

X = tf.reshape(nodeFeatures_train[:, :, 0], shape=(-1,
                                                   tf.shape(nodeFeatures_train[:, :, 0])[0], tf.shape(nodeFeatures_train[:, :, 0])[1]))
W = tf.reshape(weightedAdjacency, shape=(-1,
                                         tf.shape(weightedAdjacency)[0], tf.shape(weightedAdjacency)[1]))
model.fit(
    [X, W],
    epochs=epochs,
    shuffle=False
)
