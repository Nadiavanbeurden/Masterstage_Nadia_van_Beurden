from keras import layers
import numpy as np
import tensorflow as tf

class PositionalEncoding(layers.Layer):
    """
    Generates the sum of the positional encoding (as defined by Vaswani et al.) and the word embeddings
    """

    def __init__(self, vocab_size: int, max_input_size: int, d_model: int) -> None:
        super(PositionalEncoding, self).__init__()
        position_embedding_matrix = self.get_position_encoding(max_input_size, d_model)
        self.position_embedding_layer = layers.Embedding(input_dim=max_input_size, output_dim=d_model, weights=[position_embedding_matrix], trainable=False)

    def get_position_encoding(self, seq_len: int, d: int, n: int=10000) -> np.ndarray[np.float64]:
        "Positional encoding scheme as defined by Vaswani et al."
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P
    
    def call(self, inputs : tf.Tensor, word_embeddings: tf.Tensor=None):
        batch_size = tf.shape(inputs)[0] 
        seq_length = tf.shape(inputs)[1]

        position_indices = tf.range(seq_length)
        embedded_indices = self.position_embedding_layer(position_indices)
        return word_embeddings + embedded_indices
