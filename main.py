import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Input, Embedding, Concatenate
from tensorflow.keras.models import Model

class ClassToken(Layer):
    def __init__(self):
        super(ClassToken, self).__init__()

    def build(self, input_shape):
        # Initialize the class token variable with a normal distribution
        self.cls_token = self.add_weight(
            shape=(1, 1, input_shape[-1]),
            initializer=tf.random_normal_initializer(),
            trainable=True,
        )

    def call(self, inputs):
        # Create a tensor that replicates the class token for each item in the batch
        batch_size = tf.shape(inputs)[0]
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.cls_token.shape[-1]])
        cls_tokens = tf.cast(cls_tokens, dtype=inputs.dtype)
        return cls_tokens

def mlp_block(inputs, config):
    # Multi-layer perceptron block with two dense layers and dropout
    x = Dense(config["mlp_dim"], activation="gelu")(inputs)
    x = Dropout(config["dropout_rate"])(x)
    x = Dense(config["hidden_dim"])(x)
    x = Dropout(config["dropout_rate"])(x)
    return x

def transformer_encoder_block(inputs, config):
    # Transformer encoder block with layer normalization, multi-head attention, and MLP
    x = LayerNormalization()(inputs)
    attention_output = MultiHeadAttention(num_heads=config["num_heads"], key_dim=config["hidden_dim"])(x, x)
    x = Add()([attention_output, inputs])  # Residual connection

    x = LayerNormalization()(x)
    mlp_output = mlp_block(x, config)
    x = Add()([mlp_output, x])  # Residual connection

    return x

def VisionTransformer(config):
    # Define the input shape for the model
    input_shape = (config["num_patches"], config["patch_size"] * config["patch_size"] * config["num_channels"])
    inputs = Input(shape=input_shape)

    # Create patch embeddings
    patch_embeddings = Dense(config["hidden_dim"])(inputs)

    # Create positional embeddings
    positions = tf.range(start=0, limit=config["num_patches"], delta=1)
    position_embeddings = Embedding(input_dim=config["num_patches"], output_dim=config["hidden_dim"])(positions)
    embeddings = patch_embeddings + position_embeddings

    # Add the class token to the embeddings
    cls_token = ClassToken()(embeddings)
    x = Concatenate(axis=1)([cls_token, embeddings])

    # Apply multiple transformer encoder layers
    for _ in range(config["num_layers"]):
        x = transformer_encoder_block(x, config)

    # Final layer normalization and dense layer for classification
    x = LayerNormalization()(x)
    x = x[:, 0, :]  # Extract the output corresponding to the class token
    x = Dropout(0.1)(x)
    outputs = Dense(10, activation="softmax")(x)

    # Create the Keras model
    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    # Define the configuration for the Vision Transformer
    config = {
        "num_layers": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "dropout_rate": 0.1,
        "num_patches": 256,
        "patch_size": 32,
        "num_channels": 3,
    }

    # Instantiate and summarize the Vision Transformer model
    vit_model = VisionTransformer(config)
    vit_model.summary()
