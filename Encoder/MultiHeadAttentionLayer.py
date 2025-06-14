import DotProductAttention
import tensorflow as tf
from keras import layers
from numpy import random

class MultiHeadAttentionLayer(layers.Layer): 
    """
    Computes the multi head attention score
    """
    def __init__(self, num_heads: int, d_model: int, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__()
        self.heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model devided by num_heads must result in an integer" 

        # Initialize weight matrices
        self.dim = int(d_model/num_heads)
        self.W_q = layers.Dense(self.dim)
        self.W_k = layers.Dense(self.dim)
        self.W_v = layers.Dense(self.dim)
        self.W_o = layers.Dense(d_model)

        self.attention = DotProductAttention.DotProductAttention()

    def reshape_tensor(self, x: tf.Tensor, heads: int, flag: int): 
        if flag: 
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            # From Q, K, V to reshaped projection matrix
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], heads, -1))
            x = tf.transpose(x, perm=(0,2,1,3))
        else: 
            # Reverting the reshaping and transposing operations:  (batch_size, seq_length, d_model) 
            # From reshaped projection matrix to original Q, K, V
            x = tf.transpose(x, perm = (0,2,1,3))
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], -1))
        return x      
    
    def __call__(self, queries: tf.Tensor, keys: tf.Tensor, values: tf.Tensor, mask: tf.Tensor=None): 
        # Rearrange the queries to be able to compute all heads in parallel
       
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)

        # Compute the multi-head output.
        o_reshaped = self.attention(queries=q_reshaped, keys=k_reshaped, values=v_reshaped, num_heads=self.heads, d_model=self.d_model, mask=mask)

        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)

        return self.W_o(output)  

