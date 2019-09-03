import sys
import numpy as np
from nn_interpolate_gui import InterpolateApp
from midiBuffer import midiBuffer
from midi2numpy import dm2
from util import print_drums, Gauss, Clip, load_GM_data, load_model

        
def f(dt):
    """This callback function will run once per time-step, ie once per
    beat. It gets the current output of the interpolation app, maps it
    through a neural network to get a drum pattern, and plays it. It
    also sends a visualisation of the drum pattern to the app.

    dt is a delta-time variable but we don't use it.

    """
    
    global t

    # get z from the app
    z = np.array(app.interp.interp.outputs).reshape((1, latent_dim))

    # map from Uniform [0, 1] to a Gaussian in [-10, 10]
    z = Gauss(z, 0, 2)
    z = Clip(z, -10, 10) # unlikely, but clip to avoid an infinity

    # decode from z-space to x-space
    x = decode(z)
    assert np.all(x >= 0.0)
    assert np.all(x <= 1.0)

    # convert velocity values to MIDI (integer, [0, 127])
    x = x.reshape((s0, s1))
    x *= 127
    x = x.astype('uint8')

    # update the app's visual display of the drum grid
    app.interp.drum_grid.set_data(t, x)

    # play midi
    for track in range(s0):
        if x[track, t] > 0:
            b.playChord([dm2[track]], vel=x[track, t], dur=500)

    # update time-step
    t += 1
    t %= s1


if __name__ == "__main__": 

    # load the neural network and initialise MIDI
    model_filename = sys.argv[1]
    s0, s1 = 9, 64
    encode, decode, recon, latent_dim = load_model(model_filename, s0, s1)
    n = latent_dim // 2
    b = midiBuffer(device=[], verbose=True)
    t = 0

    # start the app
    bpm = 90
    resolution = 4
    tps = bpm * resolution / 60
    app = InterpolateApp(n, f, 1.0/tps)
    app.run()
    b.close()
