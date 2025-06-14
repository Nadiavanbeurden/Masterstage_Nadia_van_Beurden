from keras import layers, Model
import tensorflow as tf

from MultiHeadAttentionLayer import MultiHeadAttentionLayer
from FeedForwardLayer import FeedForwardLayer

class EncoderLayer(Model): 
    """
    A class to generate the output of a single encoding layer.

    Attributes
    ----------
        h : int
            Number of heads in the multi-head-attention layer
        drop_rate : float between 0 and 1
            Dropout rate in dropout layer
        d_model : int
            Dimension of the model (= dimension of hidden layer)
        d_ff : int
            Dimension of the feed forward layer
    """

    def __init__(self, num_heads: int, drop_rate: float, d_model: int, d_ff: int, **kwargs) -> None: 
        super(EncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.drop_rate = drop_rate
        self.d_model = d_model
        self.d_ff = d_ff

        self.mha = MultiHeadAttentionLayer(num_heads, d_model)
        self.dropout1 = layers.Dropout(drop_rate)
        self.normalize1 = layers.LayerNormalization()
        self.feedforward = FeedForwardLayer(d_ff, d_model)
        self.dropout2 = layers.Dropout(drop_rate)
        self.normalize2 = layers.LayerNormalization()
        
    def call(self, x: tf.Tensor, padding_mask: tf.Tensor, training:bool) -> tf.Tensor: 
        # X is the output of the embedding/encoding and also the input of the mha (Q,k,V)
        # Multi-head attention layer
        attention_outp = self.mha(queries=x, keys=x, values=x, mask=padding_mask)
        
        # Add a dropout layer
        dropout1_outp = self.dropout1(attention_outp, training=training)

        # Add a add&norm layer
        addnorm_outp = self.normalize1(x + dropout1_outp) 

        # Add a fully connected layer
        ff_output = self.feedforward(addnorm_outp)

        # Add a second dropout layer
        dropout2_outp = self.dropout2(ff_output, training=training)

        # Add a final add&norm layer
        output = self.normalize2(addnorm_outp + dropout2_outp)
        return output