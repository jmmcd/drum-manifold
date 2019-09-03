"""

A Python script which generates a shell script for executing many training
runs.

It also provides functions for analysing the results.

"""


from itertools import product
from pathlib import Path

# sets of hyperparameter values, giving 96 configs in all
latent_dim_v = [2, 10]
loss_v = ["BCE"]
Lambda_v = [0.01, 0.03, 0.1]
dropout_rate_v = [0.0, 0.2]
conv_layers_v = [2]
filter_expansion_v = [1, 2]
kernel_size0_v = [3, 9]
kernel_size1_v = [3, 4]


mode = "run_training"
mode = "analyse_results"
if mode == "analyse_results":
    results = {}


def process_file(filename):
    txt = open(filename)
    L = next(txt)
    assert L.startswith("exp1")
    exp1 = tuple(map(float, next(txt).split()))
    
    L = next(txt)
    assert L.startswith("exp2")
    exp2 = tuple(map(float, next(txt).split()))

    L = next(txt)
    assert L.startswith("exp2")
    exp2a = tuple(map(float, next(txt).split()))

    L = next(txt)
    assert L.startswith("exp3")
    exp3 = tuple(map(float, next(txt).split()))

    return exp1, exp2, exp2a, exp3
    

for config in product(
        latent_dim_v, loss_v, Lambda_v,
        dropout_rate_v, conv_layers_v,
        filter_expansion_v,
        kernel_size0_v, kernel_size1_v
):
    latent_dim, loss, Lambda, dropout_rate, conv_layers, filter_expansion, kernel_size0, kernel_size1 = config
        
    dir = "../results/vae_conv/" + "_".join(map(str, config))

    if mode == "run_training":
        print("mkdir -p " + dir)

        if Path(dir + "/vae_conv_keras_drums.h5").is_file():
            print("# skipping!")
            continue

        print("#", latent_dim, loss, Lambda, dropout_rate, conv_layers, filter_expansion, kernel_size0, kernel_size1)
        
        s = """python vae_conv_keras_drums.py --latent_dim %d --loss %s --Lambda %f --dropout_rate %f --conv_layers %d --filter_expansion %d --kernel_size0 %d --kernel_size1 %d""" % config
        print(s)

        s = "mv training_vae_conv_keras_drums.csv vae_conv_keras_drums.h5 vae_conv_keras_drums_args.txt %s" % dir
        print(s)
        

        s = """python numerical_exps.py """ + dir + "/vae_conv_keras_drums.h5 > " + dir + "/results.txt && mv exp4_*.png " + dir

        print(s)

    else:

        results[config] = process_file(dir + "/results.txt")
        

"""
exp1: norm of dx/dz with delta_method random : min, mean, max, var
exp2: mean of dx/dz with delta_method fixed : min, max, norm
exp2: mean of dx/dz with delta_method fixed : min, max, norm
exp3: RMSE of x', x with x_source test : min, mean, max, var
"""



# the following was by visual inspection of the distributions from exp4:
degenerate_data = open("../results/vae_conv/degenerate.csv")
degenerate = {}
for line in degenerate_data:
    config, result = line.strip().split(",")
    degenerate[config] = result

if mode == "analyse_results":
    print(r" & & & & & \multicolumn{4}{c|}{$z=%d$} & \multicolumn{4}{c}{$z=%d$} \\" % tuple(latent_dim_v))    
    print(r"$\lambda$ & $d$ & $f$ & $k_0$ & $k_1$ & degen & mean $\|dx/dz\|$ & max mean $dx/dz$ & mean recon loss & degen & mean $\|dx/dz\|$ & max mean $dx/dz$ & mean recon loss \\")
    print(r"\hline")

    did_config = False
    gap_count = 0
    for config in product(
            loss_v, Lambda_v,
            dropout_rate_v, conv_layers_v,
            filter_expansion_v,
            kernel_size0_v, kernel_size1_v,
            latent_dim_v
    ):
        config = config[-1:] + config[:-1] # reorder because we take z last
        degen_config = "_".join(map(str, config))
        latent_dim, loss, Lambda, dropout_rate, conv_layers, filter_expansion, kernel_size0, kernel_size1 = config
        data = (Lambda, dropout_rate, filter_expansion, kernel_size0, kernel_size1)

        # horrible Latex table munging
        
        if not did_config:
            print(r"%.2f & %.1f & %d & %d & %d & " % data, end="")

        data = (degenerate[degen_config], results[config][0][1], results[config][1][2], results[config][3][1])
        print(r"%s & %.3f & %.3f & %.3f " % data, end="")

        if not did_config:
            print(r" & ", end="")

        if did_config:
            print(r"\\", end="")

        gap_count += 1
        if gap_count == 16:
            print(r"[0.1cm]")
        elif gap_count % 2 == 0:
            print("")
        gap_count %= 16
        did_config = not did_config
              
