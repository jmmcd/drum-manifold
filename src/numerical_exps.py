"""

Numerical experiments.

Consider our encoder as a mapping z = z(x) and decoder as a mapping x = x(z).

Because some of the quantities we are looking at have quite similar notation
we just names exp1, ... exp8 for the functions. They are defined as below.

1. How large is dx/dz? Its variance? It should be small for "locality" of the mapping.
1a. TODO we could ask the same question about dz/dx
2. For a fixed non-axis aligned perturbation in z, is there any consistency in delta x?
2a. For a fixed axis-aligned perturbation in z, same question. This checks whether interpretation is "rotated". 
2b. TODO For a fixed axis-aligned perturbation in z while holding other variables constant, ie all along one axis, is there a fixed delta-x? This checks whether interpretation is consistent in this more limited sense.
3. How good is reproduction x -> x'
4. How are the training and test sets distributed in the z-space?
5. TODO For random z, decode it to x', and find the nearest training point x. How close are they? This tests whether the network is mostly just memorising the training data.
6. How do outputs differ over the early epochs of training?
7. What does an interpolation look like?
8. Plot loss curves over time.
"""


import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance
from midi2numpy import dm2, write_image
from util import print_drums, Gauss, Clip, load_GM_data, load_model, norm, RMSE


def main():
    reps = 3000
    exp1(reps, delta_method="random")
    exp2(reps, delta_method="fixed", fixed_delta=get_positive_delta())
    exp2(reps, delta_method="fixed", fixed_delta=rand_vec_axis_aligned(0))
    exp3(reps, x_source="test")
    reps = 1500
    # exp4(reps, "train")
    exp4(reps, "test")
    exp6()
    exp7()
    exp8()

filename = "../data/Groove_Monkee_Mega_Pack_GM.npy"
(x_train, y_train), (x_test, y_test) = load_GM_data(filename)
s0, s1 = 9, 64

model_filename = sys.argv[1]

encode, decode, recon, latent_dim = load_model(model_filename, s0, s1)

def get_positive_delta():
    delta = np.ones((1, latent_dim)) # a "diagonal" vector, ie all axes=1
    delta /= norm(delta) # now a vector of unit length 
    delta *= L # now of length L
    return delta
    
def rand_vec_given_length():
    delta = np.random.normal(size=(1, latent_dim))
    delta /= norm(delta) # now a vector of unit length
    delta *= L # now a vector of length L
    return delta

def rand_vec_axis_aligned(axis=None):
    delta = np.zeros((1, latent_dim))
    if axis is None:
        axis = random.randrange(latent_dim)
    delta[0, axis] = L
    return delta
    
def dx_dz(z, delta):
    x = decode(z).reshape((s0, s1)) # [0, :, :, 0] # shape is (1, 9, 64, 1)
    x_ = decode(z + delta).reshape((s0, s1)) # [0, :, :, 0] # shape is (1, 9, 64, 1)
    return x_ - x

def random_z():
    z = np.random.random((1, latent_dim)) # simulate values coming from UI
    z = Gauss(z, 0, 2) # map to a wide Gaussian and clip at (-10, 10)
    z = Clip(z, -10, 10)
    return z

def random_x():
    x = np.random.random((s0, s1))
    return x

def random_x_from_training_data():
    return random.choice(x_train)

def random_x_from_test_data():
    return random.choice(x_test)

def cosine_dist(x, y):
    return scipy.spatial.distance.cosine(x.flatten(), y.flatten())

def get_delta(delta_method="random", fixed_delta=None):
    if delta_method == "random":
        return rand_vec_given_length()
    elif delta_method == "fixed":
        return fixed_delta

def exp1(reps, delta_method="random", fixed_delta=None):
    results = np.zeros(reps)
    for i in range(reps):
        z = random_z()
        delta = get_delta(delta_method, fixed_delta)
        results[i] = norm(dx_dz(z, delta))
    print("exp1: norm of dx/dz with delta_method", delta_method, ": min, mean, max, var")
    print(np.min(results), np.mean(results), np.max(results), np.var(results))

def exp2(reps, delta_method="random", fixed_delta=None):
    results = np.zeros((reps, s0, s1))
    for i in range(reps):
        z = random_z()        
        delta = get_delta(delta_method, fixed_delta)
        results[i, :, :] = dx_dz(z, delta)
    dx_mean = np.mean(results, axis=0)
    print("exp2: mean of dx/dz with delta_method", delta_method, ": min, max, norm")
    print(np.min(dx_mean), np.max(dx_mean), norm(dx_mean))

def exp3(reps, x_source="random"):
    results = np.zeros(reps)
    for i in range(reps):
        if x_source == "random":
            x = random_x()
        elif x_source == "train":
            x = random_x_from_training_data()
        elif x_source == "test":
            x = random_x_from_test_data()
        else:
            raise ValueError
        x = x.reshape((1, s0, s1, 1))
        x_ = recon(x)
        results[i] = RMSE(x, x_)
    print("exp3: RMSE of x', x with x_source", x_source, ": min, mean, max, var")
    print(np.min(results), np.mean(results), np.max(results), np.var(results))


def exp4(reps, source="train"):
    results = np.zeros((reps, latent_dim))
    if source == "train":
        data = x_train
    else:
        data = x_test
    
    for i in range(reps):
        x = data[i]
        x = x.reshape((1, s0, s1, 1))
        z = encode(x)[0] # get z_mu, not z_sampling
        results[i] = z
    n = z.shape[1] // 2
    for i in range(n):
        plt.figure(figsize=(4, 4))
        plt.scatter(results[:, i*2], results[:, i*2+1], alpha=0.5, s=10)
        plt.axis("equal")
        plt.xlim((-3.5, 3.5))
        plt.ylim((-3.5, 3.5))
        plt.xlabel(f"$z_{i*2}$")
        plt.ylabel(f"$z_{i*2+1}$")
        plt.tight_layout()
        plt.savefig("exp4_" + source + ("_%d_%d" % (i*2, i*2+1)) + ".png")
    
def exp6():
    dir = "../results/vae_conv/10_BCE_0.03_0.2_2_2_3_3"
    for epoch in [0, 1, 2, 3, 4, 5, 10, 100]:
        print(epoch)
        model_filename = dir + "/vae_conv_keras_drums_epochs_%d.h5" % epoch
        encode, decode, recon, latent_dim = load_model(model_filename, s0, s1)
        z = np.zeros((1, 10))
        x = decode(z)
        write_image(x.reshape(s0, s1), dir + "/early_epochs_%d" % epoch)
 
def exp7():
    dir = "../results/vae_conv/10_BCE_0.03_0.2_2_2_3_3"
    model_filename = dir + "/vae_conv_keras_drums.h5"
    encode, decode, recon, latent_dim = load_model(model_filename, s0, s1)
    z = np.zeros((1, 10))
    for z0 in np.linspace(-3, 3, 6):
        z[:, 0] = z0
        print(z)
        x = decode(z)
        zs = ("%.1f" % z0).replace(".", "__")
        write_image(x.reshape(s0, s1), dir + "/interpolations_%s" % zs)


def exp8():
    dir = "../results/vae_conv/10_BCE_0.03_0.2_2_2_3_3"
    fname = dir + "/training_vae_conv_keras_drums.csv"
    df = pd.read_csv(fname)
    plt.figure(figsize=(4.5, 3), dpi=100)
    plt.plot(df["epoch"], df["val_KL_loss"], label="KL loss")
    plt.plot(df["epoch"], df["val_recon_loss"], label="Recon loss")
    plt.plot(df["epoch"], df["val_loss"], label="Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(dir + "/training_loss_curve.pdf")


# about the smallest change that's possible to make using a mouse. we
# would expect it to cause few large changes in drums.
L = 0.01 

    
if __name__ == "__main__":
    main()
