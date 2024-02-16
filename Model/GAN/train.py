import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from input_pipeline import dataset_tfrecord_pipeline
import time
from IPython import display
from model import make_discriminator_model, make_generator_model, train_step
from pathlib import Path
from datetime import datetime
from pathlib import Path

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M")
results = Path("results/" + formatted_datetime)
results.mkdir()

plt.rcParams['figure.dpi'] = 300

data_path = Path("../../GetData/Python/Data")
size = 50000

data_list = []
for temp in [2.0]:
    train_path = data_path / f'Data{temp:.2}.tfrecord'
    train_ds = dataset_tfrecord_pipeline(train_path, flatten=False, batch_size=size)
    for image in train_ds.take(1):
        data_list.append(image)

batch_size = 100
train_images = np.array(data_list)
train_images = np.concatenate(train_images, axis=0)
np.random.shuffle(train_images)
train_ds = tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size)

def plot_loss(disc_loss_log, gen_loss_log, epoch):
    fig, ax1 = plt.subplots()

    ax1.plot(np.asarray(disc_loss_log), color='tab:red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Discriminator Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.plot(np.asarray(gen_loss_log), color='tab:blue')
    ax2.set_ylabel('Generator Loss', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.savefig(results / f"loss_at_epoch_{epoch}.png")
    
def generate_and_save_images(model, epoch, test_input):
  predictions = tf.round(model(test_input, training=False))

  fig = plt.figure(figsize=(4,4))
  
  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0])
      plt.axis('off')
        
  plt.savefig(results / f"image_at_epoch_{epoch}.png")
    
def train(dataset, epochs, gen_loss_log, disc_loss_log):
  for epoch in range(epochs):
    start = time.time()
    for images in tqdm(dataset): train_step(images, gen_loss_log, disc_loss_log, batch_size, noise_dim, generator, discriminator, generator_optimizer, discriminator_optimizer)
    generate_and_save_images(
      generator,
      epoch,
      random_vector_for_generation
    )
    if epoch % 5 == 0:
        plot_loss(
            gen_loss_log,
            disc_loss_log,
            epoch
        )
    print(f"Time taken for epoch {epoch} is {time.time()- start} sec")

noise_dim = 100
num_examples_to_generate = 16
random_vector_for_generation = tf.random.normal([num_examples_to_generate,
                                                 noise_dim])
generator = make_generator_model()
discriminator = make_discriminator_model()
lr = 1e-4
generator_optimizer = tf.keras.optimizers.Adam(lr)
discriminator_optimizer = tf.keras.optimizers.Adam(lr)

EPOCHS=21
gen_loss_log=[]
disc_loss_log=[]

train(train_ds, EPOCHS,gen_loss_log, disc_loss_log)

generator.save(results / 'generator_model.keras')
discriminator.save(results / 'discriminator_model.keras')