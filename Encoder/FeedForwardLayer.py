from keras import layers
import tensorflow as tf

class FeedForwardLayer(layers.Layer): 
    """Generates a FeedForward layer which enherits the keras layers methods.
    A fully connected feed-forward network consists of two linear transformations 
    with a relu activation in between. The first linear transformation produces an 
    output dimensionality d_ff while the second linear transformation 
    produces an output of d_model (Attention is all you need - Vaswani et al.)"""

    def __init__(self, d_ff: int, d_model: int, **kwargs) -> None: 
        super(FeedForwardLayer, self).__init__(**kwargs)
        # Create needed layers
        self.fully_connected1 = layers.Dense(d_ff, activation='relu')
        #self.activation = layers.ReLU()
        self.fully_connected2 = layers.Dense(d_model)
    
    def __call__(self, input: tf.Tensor) -> tf.Tensor: 
        # Input is passed into the 2 fully-connected layers with activation between.
        after_first_layer = self.fully_connected1(input)
        #after_relu = self.activation(after_first_layer)
        after_second_layer = self.fully_connected2(after_first_layer)

        return after_second_layer