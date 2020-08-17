#!/opt/anaconda3/bin/python
from data_utils import load_data
import tensorflow as tf
from tensorflow import keras
from spektral.layers import GraphConv


class GLU(keras.layers.Layer):
    def __init__(self, filters=32, kernelsize=3):
        super(GLU, self).__init__()
        self.conv_layer = keras.layers.Conv1D(
            filters=filters, kernel_size=kernelsize)

    def build(self):
        self.w = self.add_weight(initializer='random_normal', trainable=True)

    def divideInHalf(self, inputs):
        half_point = inputs.shape[1] // 2
        half1 = inputs[:, :half_point, :]
        half2 = inputs[:, half_point:, :]
        return half1, half2

    def call(self, inputs):
        interMatrix = self.conv_layer(inputs)
        half1, half2 = self.divideInHalf(interMatrix)
        gating = tf.tensordot(half1, tf.nn.sigmoid(half2), axes=0)
        stacked = tf.stack([gating, gating], axis=1)
        final_tensor = tf.nn.softmax((self.w)*stacked)
        return final_tensor


class STBlock(keras.layers.Layer):
    def __init__(self, W, inputs):
        super(STBlock, self).__init__()
        self.W = W
        self.GLU = GLU(32, 3)
        self.graphconv = GraphConv(channels=inputs.shape[1])

    def call(self, inputs):
        inter1 = GLU(inputs)
        inter2 = self.graphconv([inter1, self.W])
        blockout = GLU(inter2)
        return blockout


model = keras.models.Sequential()

nodeFeatures_train, nodeFeatures_test, weightedAdjacency = load_data(
    DATASET='data-all.json', R=300, SIGMA=1, TEST_NUMBER=150)
