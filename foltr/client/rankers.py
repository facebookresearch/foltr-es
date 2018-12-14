# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import torch


class LinearRanker(nn.Module):
    def __init__(self, n_features):
        super(LinearRanker, self).__init__()
        self.fc = nn.Linear(in_features=n_features, out_features=1, bias=False)
        self.fc.weight.data.zero_()

    def forward(self, x):
        x = self.fc(x)
        return x


class TwoLayerRanker(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(TwoLayerRanker, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc1.weight.data.zero_()
        self.fc1.bias.data.zero_()

        self.fc2 = nn.Linear(in_features=n_hidden, out_features=1, bias=False)
        self.fc2.weight.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def perturb_model(model: torch.nn.Module, seed: int, noise_std: float, swap: bool) -> torch.nn.Module:
    """
    Take a torch model, applied Gaussian random noise to its params. Used by the ES optimization.
    The noise is deterministic provided the same random seed; depedning on the swap flag the noise is either added
    or subtracted, which is useful for implementing antithetic variates.
    The original model is not modified.

    :param model: Model to be perturbed.
    :param seed: Seed for the random noise.
    :param noise_std: std for the zero-mean Gaussian noise
    :param swap: If true, noise is subtracted.
    :return: A perturbed copy of the model
    """
    rnd = np.random.RandomState(seed)
    perturbed_model = copy.deepcopy(model)
    for param in perturbed_model.named_parameters():
        param_as_np = param[1].data.numpy()
        shape = param_as_np.shape
        noise = rnd.normal(np.zeros_like(param_as_np),
                           noise_std, size=shape).astype(dtype=np.float32)
        perturbed = param_as_np + noise * (-1.0 if swap else 1.0)
        param[1].data = torch.from_numpy(perturbed)
    return perturbed_model


def update_gradients(model: torch.nn.Module,
                     noise_std: float,
                     feedbacks: List['ClientMessage'],
                     inverse: bool) -> None:
    """
    Update the gradients in the model, given the client messages. To do that, it extracts the random seeds from the
        messages and re-creates the noise. Then, weights the noise by the corresponding metric values.
    :param model: The model that was perturbed
    :param noise_std: The noise std that was used in perturbations
    :param feedbacks: A list of client messages; each message has
    :param inverse: If True, the gradients estimate's sign is swapped
    :return: Nothing, updates the gradient nodes in the model directly
    """
    assert len(feedbacks) > 0
    antithetic = feedbacks[0].antithetic_metric is not None

    scaler = 1.0 / noise_std / noise_std / len(feedbacks)
    # we only use privatized metric in optimization, so let's strip everything else
    if antithetic:
        scaler *= 0.5
        feedbacks = [(m.seed, m.metric.privatized,
                      m.antithetic_metric.privatized) for m in feedbacks]
    else:
        feedbacks = [(m.seed, m.metric.privatized) for m in feedbacks]

    if inverse:
        scaler *= -1.0

    params_current = dict(model.named_parameters())
    for feedback in feedbacks:
        seed = feedback[0]
        perturbed_model = perturb_model(model, seed, noise_std, swap=False)

        params_perturbed = dict(perturbed_model.named_parameters())

        for name, var_orig in params_current.items():
            var_perturbed = params_perturbed[name]
            score = feedback[1]
            if antithetic:
                score -= feedback[2]
            estimate = (var_perturbed.data - var_orig.data) * scaler * score
            if var_orig.grad is None:
                var_orig.grad = estimate
            else:
                var_orig.grad += estimate
