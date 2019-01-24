# VAE code from http://www.cvc.uab.es/people/joans/slides_tensorflow/tensorflow_html/vae-Jan-Hendrik-Metzen.html

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from hw11.VAE import VariationalAutoencoder, train, mnist

network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=20)  # dimensionality of latent space

vae = VariationalAutoencoder(network_architecture, learning_rate=0.001, batch_size=100)
# constructor of vae starts an interactive session

train_new_model = False
if train_new_model:
    train(vae, batch_size=100, training_epochs=75)
    vae.save("models/model_75_epochs")
else:
    vae.load("models/model_75_epochs")

labels = np.argmax(mnist.test.labels, axis=1)  # convert from one-hot to integers

def generate_interpolates(equal):
    for i in range(10):
        images = mnist.test.images[labels == i]
        images2 = images if equal else mnist.test.images[labels != i]
        im1 = images[np.random.choice(len(images))]
        im2 = images2[np.random.choice(len(images2))]
        codes = vae.transform(np.array([im1, im2]))
        diff = codes[1] - codes[0]
        path = 'same' if equal else 'diff'
        plt.imsave('images/{}/{}_0.png'.format(path, i), im1.reshape(28, 28), cmap='gray', format='png')
        for j in range(1, 8):
            code = codes[0] + (j / 8) * diff
            x_code = vae.generate(np.tile(code, (100, 1)))
            plt.imsave('images/{}/{}_{}.png'.format(path, i, j), x_code[0].reshape(28, 28), cmap='gray', format='png')
        plt.imsave('images/{}/{}_8.png'.format(path, i), im2.reshape(28, 28), cmap='gray', format='png')

generate_interpolates(False)
