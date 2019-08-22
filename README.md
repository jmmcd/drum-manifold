# drum-manifold
Representation Learning for Drum Loops with Neural Networks

Support for a paper presented at CSMC 2018.

To use this, you need to install a Python MIDI library as follows: `pip install git+https://github.com/jmmcd/python-midi/`.

Then you need a MIDI drum-loop library for training data. I used a library from Groove Monkee. They have a cut-down version available for free which works. Start by pre-processing it (where `<directory>` contains the library, and this will write out images of the MIDI files plus one large numpy file):

`python midi2numpy.py convert_lib <directory>`

Then you can run the neural network:

`python vae.py`


Some examples of the trained NN being used in a GUI: 

<object width="425" height="344"><param name="movie" value="https://www.youtube.com/v/3kzbQI2LiOk&hl=en&fs=1"></param><param name="allowFullScreen" value="true"></param><embed src="https://www.youtube.com/v/3kzbQI2LiOk&hl=en&fs=1" type="application/x-shockwave-flash" allowfullscreen="true" width="425" height="344"></embed></object>

