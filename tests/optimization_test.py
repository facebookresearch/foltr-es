# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.resolve())
from foltr.client.client import ClientMessage
from foltr.client.metrics import MetricValue
from foltr.client.rankers import LinearRanker, perturb_model, update_gradients


def test_model_pertubation():
    model = LinearRanker(16)

    perturbed_model_1 = perturb_model(model, seed=7, noise_std=1.0, swap=False)
    perturbed_model_2 = perturb_model(model, seed=7, noise_std=1.0, swap=False)

    # the original model is initialized with zeros and should remain like that
    for param in model.parameters():
        assert torch.norm(param) == 0

    params_1 = dict(perturbed_model_1.named_parameters())
    params_2 = dict(perturbed_model_2.named_parameters())

    # the perturbed models should actually go away from zeros; the perturbations with the same seed should be
    # identical
    for name in params_1:
        assert torch.norm(params_1[name]) > 0
        assert params_1[name].equal(params_2[name])

    # antithetic change from zero would be negative of the original change
    anti_perturbed_model = perturb_model(
        model, seed=7, noise_std=1.0, swap=True)
    for name, param in anti_perturbed_model.named_parameters():
        assert param.mul(-1).equal(params_1[name])

    # with a different seed it should be a different model
    another_perturbed_model = perturb_model(
        model, seed=17, noise_std=1.0, swap=False)
    for name, param in another_perturbed_model.named_parameters():
        assert not param.equal(params_1[name])


def test_optimization():
    """
    We define loss of the model as the squared distance of its parameters from the point (1.0, 1.0, ...., 1.0)
    and then optimize parameters of this model to minimize its loss. To do so, we write "fake" ClientMessages
    that encode the loss.
    """

    def get_loss(m: torch.nn.Module) -> float:
        loss = 0.0
        for param in m.parameters():
            loss += torch.norm(param - 1.0).pow(2.0)
        return loss.item()

    n_dim = 8
    model = LinearRanker(n_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    std = 1e-1

    loss_initial = get_loss(model)
    loss_expected = np.power((np.zeros(n_dim) - 1.0), 2.0).sum()
    assert np.abs(loss_initial - loss_expected) < 1e-3

    batch_size = 50

    for epoch in range(200):
        model.zero_grad()
        messages = []
        batch_loss = 0.0
        for step in range(batch_size):
            seed = step + 1000 * epoch,
            perturbed_model = perturb_model(
                model, seed=seed, noise_std=std, swap=False)
            model_loss = get_loss(perturbed_model)
            batch_loss += model_loss

            # to make sure that non-privatized metric is never used
            metric = MetricValue(privatized=model_loss, non_privatized=0.0)
            messages.append(ClientMessage(
                seed, metric=metric, antithetic_metric=None))

        update_gradients(model, noise_std=std,
                         feedbacks=messages, inverse=False)
        optimizer.step()

    # not too good as an optimizer =D
    assert batch_loss / batch_size < 1e-1


def test_grad():
    """
    Do backprop grad and the numerical grad (via ES) coincide?
    """
    n_dim = 2
    features = torch.zeros(n_dim).normal_(1.0)

    # model_1 contains back-proped gradient
    model_1 = LinearRanker(n_dim)
    model_1.zero_grad()
    model_1.forward(features).backward()

    # model_2 has ES-based estimates of the grad
    model_2 = LinearRanker(n_dim)
    batch_size = 10000
    std = 1e-2
    messages = []
    for step in range(batch_size):
        seed = step
        perturbed_model = perturb_model(
            model_1, seed=seed, noise_std=std, swap=False)
        model_loss = perturbed_model(features).item()
        # to make sure that non-privatized metric is never used
        metric = MetricValue(privatized=model_loss, non_privatized=0.0)
        messages.append(ClientMessage(
            seed, metric=metric, antithetic_metric=None))

    update_gradients(model_2, noise_std=std, feedbacks=messages, inverse=False)

    params_1 = dict(model_1.named_parameters())

    for name, param in model_2.named_parameters():
        grad_1 = params_1[name].grad
        grad_2 = param.grad

        # not too precise, huh
        assert torch.norm(grad_1 - grad_2) < 0.2
