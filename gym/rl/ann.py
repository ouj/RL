import tensorflow as tf

# Hidden layers
def create_hidden_layers(x, sizes=(100,), activation=tf.nn.relu):
    for size in sizes:
        layer = tf.keras.layers.Dense(units=size, activation=activation)
        x = layer(x)
    return x

#%% Helper functions
def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]