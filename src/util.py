import numpy as np
from scipy.special import erfinv
import time
from argparse import Namespace

# https://stackoverflow.com/questions/45296586/how-to-convert-uniform-normality-variables-in-python
Gauss = lambda x, mu, sigma: mu + np.sqrt(2)*sigma*erfinv(2*x-1) 
Clip = lambda x, lb, ub: np.minimum(ub, np.maximum(x, lb))



norm = np.linalg.norm


def RMSE(x, y):
    return norm(x - y)

def print_drums(x):
    s0, s1 = x.shape
    for track in range(s0-1, -1, -1):
        print(" ".join(str(int(x[track, t] // 13)) for t in range(s1)))
    print("")

def play_drums(midibuffer, x, dm2):
    s0, s1 = x.shape
    for t in range(s1):
        for track in range(s0):
            midibuffer.playChord([dm2[track]], vel=x[track, t], dur=500)
        time.sleep(1.0/6.0)




# DRUMS dataset: retain in np format
def load_GM_data(filename):
    # load, convert to float32, convert to range [0, 1]
    X = np.load(filename).astype('float32')
    X /= np.max(X)

    # shuffle
    rng_state = np.random.get_state()
    np.random.seed(0)
    X = np.random.permutation(X)
    np.random.set_state(rng_state)

    # split
    train_max = round(0.9 * X.shape[0])
    x_train = X[:train_max]
    x_test  = X[train_max:]
    y_train = None
    y_test  = None
    return (x_train, y_train), (x_test, y_test)

    
def load_model(model_filename, s0, s1):
    """Load a model which has already been trained.

    NB we will be calling .predict() on the model. Keras will set
    learning_phase correctly to test mode, as stated here: https://github.com/tensorflow/tensorflow/issues/11336#issuecomment-315898065
    This is important because we have Dropout. """
    
    if model_filename.endswith(".pt"):
        # pytorch
        import torch
        from vae import VAE
        device = torch.device("cpu")
        model = VAE(s0, s1, 10, 400, 1.0, "BCE", "ReLU").to(device)
        model.eval() # tell the model we are not in training mode
        model.load_state_dict(torch.load(model_filename))
        decode = lambda z: model.decode(torch.tensor(z, dtype=torch.float).flatten()).detach().numpy()
        encode = lambda x: model.encodedecode(torch.tensor(x, dtype=torch.float).flatten()).detach().numpy()
        recon = lambda x: model.forward(torch.tensor(x, dtype=torch.float).flatten()).detach().numpy() # TODO check this
        latent_dim = 10

    elif model_filename.endswith(".h5"):
        import vae_conv_keras_drums
        config_filename = model_filename.replace(".h5", "_args.txt")
        s = open(config_filename).read()
        args = eval(s)
        
        # latent_dim, loss, Lambda, dropout_rate, conv_layers, filter_expansion, kernel_size0, kernel_size1 = config.split("_")
        
        # config = model_filename.split("/")[-2]
        
        # @dataclass
        # class Namespace:
        #     Lambda: float=0.01
        #     batch_size: int=128
        #     conv_layers: int=2
        #     dense_layer_size: int=16
        #     dropout_rate: float=0.2
        #     epochs: int=100
        #     filter_expansion: int=2
        #     filters: int=16
        #     kernel_size0: int=9
        #     kernel_size1: int=4
        #     latent_dim: int=10
        #     loss: str='BCE'
        #     stride0: int=1
        #     stride1: int=1
        #     weights: np.array=np.array([[]])
        
        # args = Args()
        # args.latent_dim = int(latent_dim)
        # args.loss = loss
        # args.Lambda = float(Lambda)
        # args.dropout_rate = float(dropout_rate)
        # args.conv_layers = int(conv_layers)
        # args.filter_expansion = int(filter_expansion)
        # args.kernel_size0 = int(kernel_size0)
        # args.kernel_size1 = int(kernel_size1)
        # args.stride0 = 1
        # args.stride1 = 1
        # args.filters = 16
        # args.dense_layer_size = 16
        # args.epochs = 100
        # args.batch_size = 128

        input_shape = (9, 64, 1) # FIXME
        inputs, outputs, z_mean, z_log_var, encoder, decoder, vae = vae_conv_keras_drums.make_model(input_shape, args)

        vae.load_weights(model_filename)
        encode = encoder.predict
        decode = decoder.predict
        recon = vae.predict

    else:
        raise

    return encode, decode, recon, int(args.latent_dim)
