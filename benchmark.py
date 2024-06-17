# %%
import torchvision
import torch.utils.data
import jax, optax
from jax import nn, vmap
from jax import numpy as jnp
from jax import random as jrandom
import equinox as eqx
from layers import FlashKAN

from time import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
# %%
key = jrandom.key(23)
# %% MNIST data loaders
batch = 200
transform = torchvision.transforms.ToTensor()

train_data = torchvision.datasets.MNIST("./Data", train=True, download=True,
                                        transform=transform)
train_data, val_data = torch.utils.data.random_split(train_data, [5/6, 1/6])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch,
                                          shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch,
                                          shuffle=True)

test_data = torchvision.datasets.MNIST(root='./Data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch,
                                         shuffle=False)

criterion = lambda logits, labels: jnp.mean(
    optax.losses.softmax_cross_entropy_with_integer_labels(logits, labels))
metric = lambda out, labels: jnp.mean(jnp.float32(jnp.argmax(out,1) == labels))
# %%
# class MultiLayerKAN(eqx.Module):
#     layers: list
    
#     def __init__(self, layers):
#         super().__init__()
#         self.layers = layers
        
#     def __call__(self, x: jax.Array):
#         for layer in self.layers:
#             x = layer(x)
#         return x
            
def create_model(w, G, key):
    KANLayer = FlashKAN
    keys = jrandom.split(key, 2)
    
    # layers = [
    #     jnp.ravel,
    #     KANLayer(28*28, w, G, key=keys[0]),
    #     KANLayer(w, 10, G, key=keys[1])
    # ]
    # net = MultiLayerKAN(layers)
    net = eqx.nn.Sequential([
        eqx.nn.Lambda(jnp.ravel),
        KANLayer(28*28, w, G, key=keys[0]),
        KANLayer(w, 10, G, key=keys[1])
    ])
    opt_state = opt.init(eqx.filter(net, eqx.is_array))
    return net, opt_state

opt = optax.adam(0.001)
# %%
metric = eqx.filter_jit(metric)

@eqx.filter_jit
def loss_fn(net, inputs, labels, key):
    net_call = vmap(net)
    return criterion(net_call(inputs), labels)

@eqx.filter_jit
def train_step(net, opt_state, inputs, labels, key):
    net_call = vmap(net)
    outputs = net_call(inputs)
    loss, gs = eqx.filter_value_and_grad(loss_fn)(net, inputs, labels, key)
    updates, opt_state = opt.update(gs, opt_state, net)
    net = eqx.apply_updates(net, updates)
    
    return net, opt_state, loss, outputs

@eqx.filter_jit
def test_eval(net, inputs, labels, key):
    net_call = vmap(net)
    outputs = net_call(inputs)
    return criterion(outputs, labels), metric(outputs, labels)

def train_model(net, opt_state, epochs=30, *, key):
    history = {"epoch": [],
               "time": [],
               "train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []}
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        net = eqx.nn.inference_mode(net, False)
        running_loss = 0.
        running_acc = 0.
        t0 = time()
        
        
        for i, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = jnp.asarray(inputs.numpy()), jnp.asarray(labels.numpy())
            skey, key = jrandom.split(key)
            net, opt_state, loss, outputs = train_step(net, opt_state, inputs,
                                            labels, skey)
            acc = metric(outputs, labels)
            
            # log statistics
            running_acc += acc.item()
            running_loss += loss.item()
        t1 = time()
        
        
        history["epoch"].append(epoch)
        history["time"].append(t1-t0)
        history["train_loss"].append(running_loss / i)
        history["train_acc"].append(running_acc / i)
        
        running_loss = 0.
        running_acc = 0.
        
        net = eqx.nn.inference_mode(net, True)
        for j, (inputs, labels) in enumerate(val_loader, 1):
            inputs, labels = jnp.asarray(inputs), jnp.asarray(labels)
            skey, key = jrandom.split(key)
            loss, acc = test_eval(net, inputs, labels, skey)
            running_acc += acc.item()
            running_loss += loss.item()
        
        history["val_loss"].append(running_loss / j)
        history["val_acc"].append(running_acc / j)
    
    return net, history
# %%
Gs = [50]
ws = [32]
logs_flashkan = []

for G, w in product(Gs, ws):
    skey, key = jrandom.split(key)
    net, opt_state = create_model(w, G, skey)
    
    skey, key = jrandom.split(key)
    net, hist = train_model(net, opt_state, 10, key=skey)
    logs_flashkan.append(hist)
# %%