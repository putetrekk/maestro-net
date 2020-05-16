from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Attention
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow_addons as tfa


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class TFAttentionModel(Model):
    Name = 'Attention Model v2 with TF attention layer'

    def __init__(self, sequence_length, total_words, *args, **kwargs):
        input_layer = Input(shape=[sequence_length], batch_size=1, dtype='int32')

        embedding = Embedding(input_dim=total_words,
                              output_dim=10,
                              input_length=sequence_length,
                              batch_size=1)(input_layer)

        # the encoder (lstm_layer)
        #   - encoder_outputs : [max_time, batch_size, num_units]
        #   - encoder_state   : [state_h, state_c]
        #       * state_h --- The Hidden State
        #       * state_c --- The Cell   State
        encoder_outputs, state_h, state_c = LSTM(150,
              dropout=0.2,
              return_sequences=True,
              return_state=True,
              recurrent_initializer='glorot_uniform')(embedding)

        attention_result, attention_weights = BahdanauAttention(50)(encoder_outputs, state_h)

        lstm = LSTM(100, recurrent_initializer='glorot_uniform')(attention_result)

        output = Dense(total_words, activation='softmax')(lstm)

        super().__init__(inputs=input_layer, outputs=output, *args, **kwargs)
        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

        print(self.summary())