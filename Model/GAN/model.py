import tensorflow as tf


def make_generator_model(lattice_size = 32):
    assert lattice_size % 4 == 0
    hidden_size = lattice_size // 4
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden_size*hidden_size*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
      
    model.add(tf.keras.layers.Reshape((hidden_size, hidden_size, 256)))
    assert model.output_shape == (None, hidden_size, hidden_size, 256) # Note: None is the batch size
    
    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, hidden_size, hidden_size, 128)  
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 2 * hidden_size, 2 * hidden_size, 64)    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation=tf.tanh))
    assert model.output_shape == (None, 4 * hidden_size, 4 * hidden_size, 1)
  
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
      
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
       
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
     
    return model


def generator_loss(generated_image):
    return tf.losses.binary_crossentropy(tf.ones_like(generated_image), generated_image)


def discriminator_loss(real_image, generated_image):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.losses.binary_crossentropy(tf.ones_like(real_image), real_image, from_logits = True)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.binary_crossentropy(tf.zeros_like(generated_image), generated_image, from_logits = True)

    total_loss = real_loss + generated_loss

    return total_loss