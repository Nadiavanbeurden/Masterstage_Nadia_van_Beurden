import tensorflow as tf
from keras import layers

class DotProductAttention(layers.Layer): 
    def __init__(self, **kwargs): 
        super(DotProductAttention, self).__init__(**kwargs)
    """
    A class to generate the output of a single attention layer.

    Methods
    ----------
        call()
            returns the attention, based on the queries, keys and values

    """
    
    def call(self, queries: tf.Tensor, keys: tf.Tensor, values: tf.Tensor, num_heads: int, d_model: int, mask: tf.Tensor=None): 
        """ 
        Computes the attention scores
        """
        # Compute scores based on  'Attention is all you need'
        d_k = d_model/num_heads
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(tf.cast(d_k, tf.float32))

        # Apply mask
        if mask is not None: 
            scores += -1e9 * mask 

        # Computing the weights by a softmax operation
        weights = tf.keras.activations.softmax(scores)
        
        # Computing the attention
        attention = tf.matmul(weights, values)

        return attention       