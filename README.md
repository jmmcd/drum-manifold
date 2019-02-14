# drum-manifold
Representation Learning for Drum Loops with Neural Networks

Support for a paper presented at CSMC 2018.

To use this, you need a MIDI drum-loop library. I used a library from Groove Monkee. They have a cut-down version available for free which works. Start by pre-processing it (where `<directory>` contains the library, and this will write out images of the MIDI files plus one large numpy file):

`python midi2numpy.py convert_lib <directory>`

Then you can run the neural network:

`python vae.py`

