import numpy as np
import pennylane as qml
import jax
from jax import numpy as jnp
from jax import grad, jit, vmap
import optax

from functools import partial
from tqdm import tqdm
from time import perf_counter
import yaml

import os
from datetime import datetime


"""
Experimental design:
 - Benchmark in terms of batch size

 - Benchmark mean execution time of training iteration per batch
 - Benchmark vmap and no vmap
 - Benchmark JIT and no JIT

 - Benchmark GPU only
 - Track memory usage (not now, maybe later)
"""

"""
Engineering:
 - [DONE] Add warm-up
 - [DONE] Update batching
 - [DONE] Add validation
 - [DONE] Save results
   [DONE]  - paramters as numpy array
   [DONE]  - benchmarking times as dict
   [DONE]  - loss+accuracy as array
 - [DONE] Measure time
 - [DONE] Make it more configurable for benchmarking

 - [DONE] Limit the number of batches for benchmarking
 - [DONE] Add hparams for experiments (cpu/gpu, vmap, jit, backend)
 - General cleanup (use more function e.g. for data loading)
 - [WONT DO] Update environment to enable parallel compilation
 - [DONE] Use CIFAR 10 dataset
 - [DONE] Use pilot executor to submit jobs
 - Save in the correct directory
"""

def training(config):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basepath_results = f"/pscratch/sd/f/fkiwit/work_classifier/j{config['jit']}_v{config['vmap']}_b{config['batch_size']}_{timestamp}"
    os.makedirs(basepath_results, exist_ok=True)
    # jax.config.update("jax_compilation_cache_dir", "jit_compiled")
    jax.config.update('jax_platform_name', config["device"])
    jax.config.update("jax_enable_x64", True)
    times = {}

    dev = qml.device("default.qubit")
    @qml.qnode(dev)
    def circuit(weights, state):
        qml.StatePrep(state, wires=range(config["n_qubits"]))

        counter = 0
        for layer_weights in weights:
            for wire in range(config["n_qubits"]):
                qml.Rot(*layer_weights[wire], wires=wire)
                counter += 1
            for wire in range(config["n_qubits"]):
                qml.CNOT([wire, (wire + 1) % config["n_qubits"]])

        return [qml.expval(qml.PauliZ(i)) for i in range(10)]


    # Define model, cost, and grad functions
    def model_base(weights, state):
        output = jnp.array(circuit(weights, state))
        return jax.nn.softmax(output)

    def cost_fn_base(weights, state, target):
        output = jnp.array(model_base(weights, state))
        return -jnp.sum(target * jnp.log(output))

    grad_fn_base = grad(cost_fn_base)

    def map_over_inputs(base_fn, param, *args):
        outputs = []
        for arg_tuple in zip(*args):
            output = base_fn(param, *arg_tuple)
            outputs.append(output)
        return jnp.stack(outputs)

    # vmap
    if config["vmap"]:
        model = vmap(model_base, in_axes=(None, 0))
        cost_fn = vmap(cost_fn_base, in_axes=(None, 0, 0))
        grad_fn = vmap(grad_fn_base, in_axes=(None, 0, 0))
    else:
        model = partial(map_over_inputs, model_base)
        cost_fn = partial(map_over_inputs, cost_fn_base)
        grad_fn = partial(map_over_inputs, grad_fn_base)

    # JIT compilation
    if config["jit"]:
        model = jit(model)
        cost_fn = jit(cost_fn)
        grad_fn = jit(grad_fn)

    # Load data
    basepath = "/global/homes/f/fkiwit/dev/data_compression/results/cifar_new"
    states_train = np.load(f"{basepath}/train_data.npy")
    states_val = np.load(f"{basepath}/test_data.npy")
    targets_train = np.load(f"{basepath}/train_labels.npy")
    targets_val = np.load(f"{basepath}/test_labels.npy")

    # Batch the data
    n_batches = len(states_train) // config["batch_size"]
    # One hot encoding
    targets_train = np.eye(10)[targets_train]
    targets_val = np.eye(10)[targets_val]
    # Create batches
    states_train_batches = np.array_split(states_train, n_batches)
    states_val_batches = np.array_split(states_val, n_batches)
    targets_train_batches = np.array_split(targets_train, n_batches)
    targets_val_batches = np.array_split(targets_val, n_batches)

    if config["n_batches"]:
        n_batches = config["n_batches"]
        states_train_batches = states_train_batches[:n_batches]
        states_val_batches = states_val_batches[:n_batches]
        targets_train_batches = targets_train_batches[:n_batches]
        targets_val_batches = targets_val_batches[:n_batches]

    # Warm-up model, cost, and grad functions
    weights = np.random.uniform(size=(config["depth"], config["n_qubits"], 3))
    print("Warming up the model, cost, and grad functions...")
    warmup_states = states_train[:config["batch_size"]]
    warmup_targets = targets_train[:config["batch_size"]]
    start = perf_counter()
    _ = model(weights, warmup_states)
    _ = cost_fn(weights, warmup_states, warmup_targets)
    _ = grad_fn(weights, warmup_states, warmup_targets)
    times["time_warmup"] = perf_counter() - start

    # Optimizer
    solver = optax.adam(1e-3)
    opt_state = solver.init(weights)
    print(f"Starting optimization with training inputs of shape {states_train.shape}")

    start = perf_counter()
    times_training_loop = []
    for epoch in range(config["n_epochs"]):
        epoch_loss = 0
        epoch_accuracy = 0
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        with tqdm(total=n_batches, desc=f"Epoch {epoch+1}/{config['n_epochs']}") as pbar:
            for batch_states, batch_targets in zip(states_train_batches, targets_train_batches):
                start = perf_counter()
                loss = jnp.mean(cost_fn(weights, batch_states, batch_targets))
                gradient = jnp.mean(grad_fn(weights, batch_states, batch_targets), axis=0)
                updates, opt_state = solver.update(gradient, opt_state, weights)
                weights = optax.apply_updates(weights, updates)
                accuracy = jnp.mean(jnp.argmax(batch_targets, axis=1) == jnp.argmax(model(weights, batch_states), axis=1))
                epoch_loss += loss
                epoch_accuracy += accuracy
                times_training_loop.append(perf_counter() - start)

                pbar.set_postfix(loss=str(loss)[:5], accuracy=str(accuracy)[:5])
                pbar.update(1)

        epoch_loss /= n_batches
        epoch_accuracy /= n_batches

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validation
        val_loss = 0
        val_accuracy = 0
        for batch_states, batch_targets in zip(states_val_batches, targets_val_batches):
            loss = jnp.mean(cost_fn(weights, batch_states, batch_targets))
            accuracy = jnp.mean(jnp.argmax(batch_targets, axis=1) == jnp.argmax(model(weights, batch_states), axis=1))
            val_loss += loss
            val_accuracy += accuracy

        val_loss /= len(states_val_batches)
        val_accuracy /= len(states_val_batches)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{config['n_epochs']} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    times["time_training"] = perf_counter() - start

    results_np = {
        "times_training_loop": np.array(times_training_loop),
        "weights": weights,
        "train_losses": np.array(train_losses),
        "train_accuracies": np.array(train_accuracies),
        "val_losses": np.array(val_losses),
        "val_accuracies": np.array(val_accuracies)
    }

    results_dict = {
        "config": config,
        "times": times
    }

    for key, value in results_np.items():
        print(f"{key}: {value.shape}")
        print(basepath_results)
        np.save(f"{basepath_results}/{key}.npy", value, allow_pickle=True)

    for key, value in results_dict.items():
        with open((f"{basepath_results}/{key}.yml"), "w") as filehandler:
            yaml.dump(value, filehandler)

if __name__ == "__main__":
    # Set hyperparameters
    config = {
        "n_qubits": 13,
        "depth": 2,
        "batch_size": 80,
        "n_batches": 10,
        "n_epochs": 2,
        "jit": True,
        "vmap": True,
        "device": "gpu"
    }
    training(config)
