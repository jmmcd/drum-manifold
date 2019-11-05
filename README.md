# drum-manifold
Representation Learning for Drum Loops with Neural Networks

Support for a paper submitted to the Springer book Artificial Intelligence in Arts, Music and Design.

To use this, you need to install a Python MIDI library as follows: `pip install git+https://github.com/jmmcd/python-midi/`.

Also Keras/Tensorflow, Kivy, PyGame:

pip install kivy
pip install pygame
pip install --upgrade pip
pip install tensorflow

Then you need a MIDI drum-loop library for training data. I used a library from Groove Monkee. They have a cut-down version available for free which works. Start by pre-processing it (where `<directory>` contains the library, and this will write out images of the MIDI files plus one large numpy file):

`python midi2numpy.py convert_lib <directory>`

Then you can run the neural network, eg:

`python vae_conv_keras_drums.py --latent_dim 2 --loss BCE --Lambda 0.010000 --dropout_rate 0.000000 --conv_layers 2 --filter_expansion 1 --kernel_size0 3 --kernel_size1 3`

Some examples of the trained NN being used in a GUI: 

[![Video showing the trained NN being used in a GUI (2D)](https://img.youtube.com/vi/3kzbQI2LiOk/0.jpg)](https://youtu.be/3kzbQI2LiOk)

[![Video showing the trained NN being used in a GUI (10D)](https://img.youtube.com/vi/7x4df0JhgQg/0.jpg)](https://youtu.be/7x4df0JhgQg)

[![Video showing the trained NN being used in a GUI (10D)](https://img.youtube.com/vi/qZxjE6fngJI/0.jpg)](https://youtu.be/qZxjE6fngJI)

[![Video showing the trained NN being used in a GUI (10D)](https://img.youtube.com/vi/BBidFxZ4IaU/0.jpg)](https://youtu.be/BBidFxZ4IaU)





Support for a previous, related paper presented at CSMC 2018 is in the CSMC_2018 directory.
