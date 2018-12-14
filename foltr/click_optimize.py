# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict, Any, NamedTuple, List
import torch
import numpy as np
from foltr.client.client import RankingClient
from foltr.client import rankers
from foltr.data.datasets import DataSplit


TrainResult = NamedTuple("TrainResult", [
                        ('batch_metrics', List[float]),
                        ('expected_metrics', List[float]),
                        ('ranker', torch.nn.Module)])


def train_uniform(params: Dict[str, Any], dataset: DataSplit) -> TrainResult:
    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_clients = params['n_clients']
    sessions_per_feedback = params['sessions_per_feedback']
    click_model = params['click_model']
    online_metric = params['online_metric']
    noise_std = params['noise_std']
    ranker = params['ranker_generator']()
    optimizer = torch.optim.Adam(ranker.parameters(), lr=params['lr'])
    antithetic = params['antithetic']

    clients = [RankingClient(dataset, ranker, seed + client_id, click_model, online_metric, antithetic, noise_std)
               for client_id in range(n_clients)]

    n_iterations = params['sessions_budget'] // n_clients // sessions_per_feedback

    batch_rewards = []
    expected_rewards = []

    for i in range(n_iterations):
        ranker.zero_grad()
        feedback = []
        for client in clients:
            f = client.get_click_feedback(sessions_per_feedback)
            feedback.append(f)
        # Those non-privatized values are solely used for plotting the performance curves
        if not antithetic:
            batch_reward = np.mean([f.metric.non_privatized for f in feedback])
        else:
            batch_reward = np.mean(
                [0.5 * (f.metric.non_privatized + f.antithetic_metric.non_privatized) for f in feedback])
        batch_rewards.append(batch_reward)

        rankers.update_gradients(ranker, noise_std, feedback, inverse=True)
        optimizer.step()
        for client in clients:
            client.update_model(ranker)

    return TrainResult(batch_metrics=batch_rewards, ranker=ranker, expected_metrics=expected_rewards)
