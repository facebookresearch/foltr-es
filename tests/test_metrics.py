# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.resolve())
from eps_estimate import privatize
from foltr.client.rankers import LinearRanker
from foltr.data.datasets import QueryDocs
from foltr.client.metrics import PrivatizedMaxRR, ExpectedMaxRR
from foltr.client.click_simulate import PERFECT_MODEL, NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL


def max_rr(clicks: np.ndarray, cutoff: int = 10) -> float:
    """
    :param clicks: Indicator vector of clicked positions
    :param cutoff: Cut-off length, documents below that would be ignored
    :return: Return 1/(position of the first click) or 0.0 if no click occurred.
    Examples:
    >>> max_rr(np.zeros(10))
    0.0
    >>> max_rr(np.array([0, 0, 1]), cutoff=2)
    0.0
    >>> max_rr(np.array([0, 0, 1]), cutoff=3)
    0.3333333333333333
    >>> max_rr(np.ones(10))
    1.0
    """
    n_docs = clicks.shape[0]
    for i in range(min(cutoff, n_docs)):
        if clicks[i] > 0:
            return 1.0 / (1.0 + i)
    return 0.0


def get_random_ranked_list(depth: int = 10) -> np.ndarray:
    # relevance labels from [0, 3)
    ranking = np.random.randint(0, 3, size=depth)
    return ranking


def get_random_click_list(depth: int = 10) -> np.ndarray:
    ranking = np.random.randint(0, 1, size=depth)
    return ranking


def test_metric_agreement() -> None:
    """
    Runs a simulation of user clicking behaviour using PERFECT, NAVIGATIONAL, and INFORMATIONAL click models and checks
    if the resulting averages agree with ExpectedMaxRR metric within +-4 std. A randomly generated result page of length
    10 is used. The simulation is run for 100,000 times.
    """
    depth = 10
    n = 20000

    ranking = get_random_ranked_list(depth)

    for model in [PERFECT_MODEL, NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL]:
        expected = ExpectedMaxRR(model).eval_ranking(ranking)

        private_mrr = PrivatizedMaxRR(1.0)
        pure_metrics = []
        privatized_metrics = []
        for i in range(n):
            random_state = np.random.RandomState(i + 7)
            clicks = model(ranking, random_state)
            pure_metrics.append(max_rr(clicks))
            privatized_metrics.append(private_mrr(clicks))

        for sample in [pure_metrics, privatized_metrics]:
            empirical_mean = np.mean(sample)
            empirical_std = np.std(sample) / np.sqrt(n)
            bound = 5.0 * empirical_std  # 5 std gives like 1e-6 prob of getting out of it
            within_tolerance = empirical_mean - bound <= expected <= empirical_mean + bound
            assert within_tolerance


def test_pMaxRR_invariants() -> None:
    """
    Checking super simple invariants of PrivatizedMaxRR:
     - with p = 1.0 we always say the truth, hence privatized and non-privatized metric values are equal
     - with p = 0.0, we always change the values, hence those values are different
     - metric is between [0, 1]
    """
    depth = 10
    non_private_metric = PrivatizedMaxRR(1.0)
    overly_private_metric = PrivatizedMaxRR(0.0)

    metrics = []
    for _ in range(10):
        clicks = get_random_click_list(depth)
        metric_value = non_private_metric(clicks)
        metrics.append(metric_value)

        # with no privatization, both values should agree
        assert metric_value.non_privatized == metric_value.privatized

        metric_value = overly_private_metric(clicks)
        metrics.append(metric_value)

        # should never say the true value, hence metric values should be different
        assert not metric_value.non_privatized == metric_value.privatized

    assert all(0 <= x.privatized <= 1 for x in metrics)
    assert all(0 <= x.non_privatized <= 1 for x in metrics)


def test_ranker_metrics() -> None:
    """
    Testing the ranker-level evaluation.
    """
    features1 = np.array(
        [[1.0, 0.0, 100.0], [2.0, 1.0, 100.0]], dtype=np.float32)
    labels1 = np.array([0.0, 2.0])

    features2 = np.zeros((2, 3), dtype=np.float32)
    labels2 = np.array([0.0, 0.0])

    one_query_data = {'qid:1': QueryDocs(
        features=features1, relevance_labels=labels1)}
    repeated_query_data = {'qid:1': QueryDocs(features=features1, relevance_labels=labels1),
                           'qid:2': QueryDocs(features=features1, relevance_labels=labels1)}

    metric_perfect = ExpectedMaxRR(PERFECT_MODEL)

    good_ranker = LinearRanker(3)
    good_ranker.fc.weight.data = torch.tensor(
        [[1.0, 0.0, 0.0]], requires_grad=False).float()
    bad_ranker = LinearRanker(3)
    bad_ranker.fc.weight.data = - \
        torch.tensor([[1.0, 0.0, 0.0]], requires_grad=False).float()

    assert metric_perfect.eval_ranker(good_ranker, one_query_data) == 1.0
    assert metric_perfect.eval_ranker(bad_ranker, one_query_data) == 0.5
    assert metric_perfect.eval_ranker(good_ranker, repeated_query_data) == 1.0
    assert metric_perfect.eval_ranker(bad_ranker, repeated_query_data) == 0.5

    two_query_data = {'qid:1': QueryDocs(features=features1, relevance_labels=labels1),
                      'qid:100500': QueryDocs(features=features2, relevance_labels=labels2)}

    assert metric_perfect.eval_ranker(good_ranker, two_query_data) == 0.5
    assert metric_perfect.eval_ranker(bad_ranker, two_query_data) == 0.25


def test_privacy_smoothing() -> None:
    """
    Make sure that the way we privatize distribution in `eps_estimate` (*) actually agrees with direct
    simulation.

    (*) The analytic estimate is done via:
     privatized_prob(f) =
            p * prob(f)  ### with prob(f) we generate `f` and with `p` we leave it as it is
                ### when a metric different from `f` is generated, with prob 1-p we do not transmit it and
                ### select from other values, out of which `f` happens with prob 1/(number_of_possibilities - 1)
                ### all these values different from `f` sum to the probability 1 - prob(f)
            + (1.0 - prob(f)) * 1/(number_of_possibilities - 1) (1-p)
    """

    true_distribution = np.array([0.3, 0.0, 0.2, 0.1, 0.4])
    p = 0.21
    n_samples = 40000

    empirical = np.zeros_like(true_distribution)
    for _ in range(n_samples):
        sampled = np.random.multinomial(1, true_distribution)
        if np.random.rand() < p:
            empirical += sampled
        else:
            uniform_excluding_sampled = 1.0 - sampled
            uniform_excluding_sampled /= uniform_excluding_sampled.sum()
            re_sampled = np.random.multinomial(1, uniform_excluding_sampled)
            empirical += re_sampled
    empirical /= empirical.sum()
    analytical = privatize(true_distribution, p)
    assert np.allclose(empirical, analytical, atol=1e-2)

