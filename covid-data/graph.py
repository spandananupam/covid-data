#!/opt/anaconda3/bin/python
from data_utils import load_data
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv1D
from keras.models import Model
from keras.optimizers import Adam
from spektral.layers import GraphConv


nodeFeatures_train, nodeFeatures_test, weightedAdjacency = load_data(
    DATASET='data-all.json', R=300, SIGMA=1, TEST_NUMBER=150)

districts = weightedAdjacency.shape[0]
time_train = nodeFeatures_train.shape[0]


class GLU(keras.layers.Layer):
    def __init__(self, filters=32, kernelsize=3):
        super(GLU, self).__init__()
        self.conv_layer = Conv1D(
            filters=filters, kernel_size=kernelsize)

    def call(self, inputs):
        interMatrix1 = self.conv_layer(inputs)
        interMatrix2 = self.conv_layer(inputs)
        gated = tf.matmul(interMatrix1, tf.nn.sigmoid(interMatrix2))
        return gated


class Block:
    def __init__(self, inputs, W, filters=32, kernel_size=3):
        self.classes = inputs.shape[1]
        self.filters = filters
        self.kernel = kernel_size
        self.W = W
        self.glu = GLU(filters, self.kernel)
        self.graph_conv = GraphConv(self.classes)

    def block(self, inputs):
        l1 = self.glu(inputs)
        l2 = self.graph_conv([l1, self.W])
        l3 = self.glu(l2)
        return l3


# Model
X_in = Input(shape=(time_train, districts))
W_in = Input(shape=(districts, districts), sparse=True)
spatialTemporal = Block(X_in, W_in, 32, 3)
l1 = spatialTemporal.block(X_in)
l2 = spatialTemporal.block(l1)

model = Model(
    inputs=[nodeFeatures_train, weightedAdjacency], outputs=l2)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()
