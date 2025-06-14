from keras import layers, Model
import tensorflow as tf
import numpy as np 

from EncoderLayer import EncoderLayer
from PositionalEncoding import PositionalEncoding

class Encoder(Model): 
    """
    Complete encoder structure
    """
    def __init__(self, vocab_size: int, max_input_size: int, num_heads: int, num_layers: int, d_model: int, d_ff: int, drop_rate: float, **kwargs) -> None: 
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.max_input_size = max_input_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.drop_rate = drop_rate

        # Initialize embedding, encoding, encoder and output layers.
        self.word_embedding_layer = layers.Embedding(vocab_size, d_model, trainable=True)
        self.pos_encoding = PositionalEncoding(vocab_size, max_input_size, d_model)
        self.dropout = layers.Dropout(drop_rate)
        self.encoder_layers = [EncoderLayer(num_heads, drop_rate, d_model, d_ff) for _ in range(num_layers)]
        self.output_layer = layers.Dense(self.vocab_size, activation='softmax')        

    def build_graph(self): 
        """ 
        Include to be able to use model.summary in train loop
        """
        input_layer = layers.Input(shape=(self.max_input_size,), name='Inputs')
        return Model(inputs=[input_layer], outputs=self.call(input_layer, True, None))

    def call(self, input_sequence: tf.Tensor, training: bool, padding_mask: tf.Tensor=None) -> tf.Tensor:

        #Generate word embeddings and positional encoding
        embedded_words = self.word_embedding_layer(input_sequence)
        pos_encoding_output = self.pos_encoding(input_sequence, word_embeddings = embedded_words)        
        
        # Add dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Add a number of endoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, padding_mask, training=training)
        
        # Add the output layer
        output = self.output_layer(x)

        return output