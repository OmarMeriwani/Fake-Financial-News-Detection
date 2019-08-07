from keras import Model
from keras.layers import  Dense
import tensorflow as tf
class Attention(Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

'''
#ATTENTION
hidden_with_time_axis = tf.expand_dims(model_output, 1)
W1 = Dense(1)
W2 = Dense(1)
from keras import backend
score = tf.nn.tanh(backend.dot( W1(model_input) ,W2(hidden_with_time_axis)))
attention_weights = tf.nn.softmax(Dense(1)(score), axis=1)
context_vector = attention_weights * model_input
context_vector = tf.reduce_sum(context_vector, axis=1)
model_output = Dense(1)(context_vector)

'''