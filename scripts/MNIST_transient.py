import numpy as np
import jax.numpy as jnp
import equinox as eqx
import optax
import argparse
import os.path
import sys
import time
import struct
import hashlib

from jax.tree_util import tree_map, tree_flatten
from jax import random
from jax.lax import scan

from architectures import AKOrN_v1, AKOrN_v2, MLP
from training_loops import transient_learning

def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-path_to_dataset": {
            "help": "path to dataset in the .npz format"
        },
        "-path_to_results": {
            "help": "path to folder where to save results"
        },
        "-learning_rate": {
            "default": 1e-4,
            "type": float,
            "help": "learning rate"
        },
        "-gamma": {
            "default": 0.5,
            "type": float,
            "help": "decay parameter for the exponential decay of learning rate"
        },
        "-N_batch": {
            "default": 100,
            "type": int,
            "help": "number of samples used to average gradient"
        },
        "-N_train": {
            "default": 60000,
            "type": int,
            "help": "number of samples in the training set"
        },
        "-N_test": {
            "default": 10000,
            "type": int,
            "help": "number of samples in the test set"
        },
        "-N_updates": {
            "default": 10000,
            "type": int,
            "help": "number of updates of the model weights"
        },
        "-N_drop": {
            "default": 10000 // 2,
            "type": int,
            "help": "number of updates after which learning rate is multiplied by chosen learning rate decay"
        },
        "-N_features": {
            "default": 100,
            "type": int,
            "help": "number of features in a hidden layer"
        },
        "-N_layers": {
            "default": 4,
            "type": int,
            "help": "number of layers"
        },
        "-D": {
            "default": 4,
            "type": int,
            "help": "dimension of oscillators"
        },
        "-N_t": {
            "default": 10,
            "type": int,
            "help": "number of integration step in Kuramoto models"
        },
        "-key": {
            "default": 14,
            "type": int,
            "help": "PRNGKey for network init and training set reshuffle"
        },
        "-N_classes": {
            "default": 10,
            "type": int,
            "help": "number of classes"
        },
        "-architecture": {
            "default": "MLP",
            "type": str,
            "choices": ["MLP", "AKOrN_v1", "AKOrN_v2"],
            "help": "architecture to use"
        },
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())
    script_name = sys.argv[0].split(".")[0]

    header = ",".join([key for key in args.keys()])
    header += ",hash,final_loss,model_size,train_accuracy,test_accuracy,training_time"
    if not os.path.isfile(f"{args['path_to_results']}/results.csv"):
        with open(f"{args['path_to_results']}/results.csv", "w") as f:
            f.write(header)

    # load MNIST
    with open(args["path_to_dataset"] + '/train-images-idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        features = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        features = jnp.array(features.reshape((size, nrows*ncols)) / 255)

    with open(args["path_to_dataset"] + '/t10k-images-idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        features_ = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        features_ = jnp.array(features_.reshape((size, nrows*ncols)) / 255)
    
    features = jnp.concatenate([features, features_], axis=0)
    
    with open(args["path_to_dataset"] + '/train-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        targets = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        targets = jnp.array(targets.reshape((size,)))
    
    with open(args["path_to_dataset"] + '/t10k-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        targets_ = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        targets_ = jnp.array(targets_.reshape((size,)))
    
    targets = jnp.concatenate([targets, targets_], axis=0)

    keys = random.split(random.PRNGKey(args["key"]), 4)
    N_features_ = [features.shape[1], args["N_features"], args["N_classes"]]

    if args["architecture"] == "AKOrN_v1":
        model = AKOrN_v1.AKOrN_v1_classification(keys[0], N_features_, args["D"], args["N_layers"])
    elif args["architecture"] == "AKOrN_v2":
        model = AKOrN_v2.AKOrN_v2_classification(keys[0], N_features_, args["D"], args["N_layers"])
    elif args["architecture"] == "MLP":
        model = MLP.MLP(keys[0], N_features_, args["N_layers"])

    model_size = sum(tree_map(lambda x: jnp.size(x) if x.dtype == jnp.float32 else 2*jnp.size(x), tree_flatten(model)[0], is_leaf=eqx.is_array))

    sc = optax.exponential_decay(args["learning_rate"], args["N_drop"], args["gamma"])
    optim = optax.lion(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    inds = random.choice(keys[1], args["N_train"], (args["N_updates"], args["N_batch"]))
    carry = [model, features, targets, opt_state, keys[2]]
    make_step_scan_ = lambda a, b: transient_learning.make_step_scan(a, b, optim, args["N_t"])

    start = time.time()
    res, losses = scan(make_step_scan_, carry, inds)
    stop = time.time()
    training_time = stop - start
    model, opt_state = res[0], res[-2]

    _, is_correct = scan(lambda a, b: transient_learning.compute_error(a, b, args["N_t"]), [model, features, targets, keys[3]], jnp.arange(features.shape[0]))
    test_accuracy = jnp.sum(is_correct[args["N_train"]:]) / is_correct[args["N_train"]:].shape[0]
    train_accuracy = jnp.sum(is_correct[:args["N_train"]]) / is_correct[:args["N_train"]].shape[0]

    # saving model, opt state and learning curve
    exp_hash = hashlib.sha256(str.encode(script_name + "".join([str(args[k]) for k in args.keys()]))).hexdigest()

    eqx.tree_serialise_leaves(f"{args['path_to_results']}/model_{exp_hash}.eqx", model)
    eqx.tree_serialise_leaves(f"{args['path_to_results']}/opt_state_{exp_hash}.eqx", opt_state)
    jnp.save(f"{args['path_to_results']}/learning_curve_{exp_hash}.npy", losses)

    data = "\n" + ",".join([str(args[key]) for key in args.keys()])
    data += f",{exp_hash},{losses[-1]},{model_size},{train_accuracy},{test_accuracy},{training_time}"

    with open(f"{args['path_to_results']}/results.csv", "a") as f:
        f.write(data)