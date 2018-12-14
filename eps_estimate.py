# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Tuple
import numpy as np
import argparse
from foltr.client.click_simulate import PERFECT_MODEL, NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL, CcmClickModel


def get_distribution(relevance_labels: List[float], click_model: CcmClickModel) -> np.ndarray:
    """
    Given a ranked list with relevance labels and a click model,
    return len(ranking) + 1 probabilities that specify the position of the first click. We have more probabilities than
    potential click positions as it could be that no click occurred.
    :param relevance_labels: numpy array with per-document relevance labels (ranked by position)
    :param click_model: CCM click model
    :return: A vector of probabilities of MaxRR values

    Example:
    >>> relevance_labels = np.zeros(10)
    >>> # no relevant documents, hence no clicks under Perfect model
    >>> # the mass of the prob distribution would be concentrated in one value
    >>> get_distribution(relevance_labels, PERFECT_MODEL)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    >>> # under a informational model, all document have some positive probability
    >>> # of being clicked
    >>> get_distribution(relevance_labels, INFORMATIONAL_MODEL)
    array([0.4       , 0.24      , 0.144     , 0.0864    , 0.05184   ,
           0.031104  , 0.0186624 , 0.01119744, 0.00671846, 0.00403108,
           0.00604662])
    >>> # when there's only one highly relevant doc, it would be always clicked under the Perfect model - again
    >>> # all mass is in a single value (but a different one)
    >>> relevance_labels = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    >>> get_distribution(relevance_labels, PERFECT_MODEL)
    array([0., 0., 0., 1., 0., 0., 0., 0., 0.])
    """
    click_relevance = click_model.click_relevance

    p_not_clicked_yet = 1.0
    n = len(relevance_labels)
    p_first_click = np.zeros(n + 1)
    for i in range(n):
        r = relevance_labels[i]
        p_click = click_relevance[r]

        p_first_click[i] = p_click * p_not_clicked_yet
        p_not_clicked_yet *= 1.0 - p_click
    p_first_click[-1] = 1.0 - p_first_click.sum()
    return p_first_click


def enumerate_distributions(depth: int, click_model: CcmClickModel, p: float) -> List[np.ndarray]:
    """
    We iterate over all possible placements of relevance scores {0, 1, 3}
    over ranked lists of depth `depth` (3**`depth` of them), get the prob. distribution of first clicks for those
    ranked lists. As there's a one-to-one mapping between the first click and the value of MaxRR, we treat those
    distributions as distributions of MaxRR metric values. Next, we apply the privatization noise and iterate over all
    pairs of metric distributions and find the pair that maximizes the privacy loss.

    :param depth: Length of the result lists considered
    :param click_model: CCM click model used to simulate clicks
    :param p: Privatization parameter, in [0, 1]
    :return: List of vectors, each characterizing metric distribution for a particular arrangement of relevance labels
            over a result list
    """
    metrics_distributions = []
    for i in range(3**depth):
        ranking = []
        for _ in range(depth):
            ranking.append(i % 3)
            i = i // 3
        distribution = get_distribution(ranking, click_model)
        distribution = privatize(distribution, p)
        metrics_distributions.append(distribution)
    return metrics_distributions


def get_most_dissimilar(distributions: List[np.ndarray]) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    """
    Finds two most "dissimilar" distributions `a` and `b` that maximize log (a[i] / b[i]) for some i
    :param distributions: List of vectors, each specifies a metric distribution
    :return: (max log-ratio over all pairs of vectors, (a, b)).

    Example:
    >>> a = np.array([0.1, 0.1, 0.1, 0.7])
    >>> b = np.array([0.8, 0.05, 0.05, 0.1])
    >>> get_most_dissimilar([a, b, a])
    (2.0794415416798357, (array([0.1, 0.1, 0.1, 0.7]), array([0.8 , 0.05, 0.05, 0.1 ])))
    >>> # which makes sense as log(8) == 2.0794415416798357
    """
    max_dissimilarity, dissimilar = None, None
    for i in range(len(distributions)):
        for j in range(i + 1, len(distributions)):
            log_ratio = np.log(
                (distributions[i][0] / distributions[j][0]))
            log_ratio = max(log_ratio.max(), -log_ratio.min())
            if max_dissimilarity is None or log_ratio > max_dissimilarity:
                max_dissimilarity = log_ratio
                dissimilar = (distributions[i], distributions[j])
    return max_dissimilarity, dissimilar


def privatize(distribution: np.ndarray, p: float) -> np.ndarray:
    """
    Given a distribution of metric values, transforms it to a distribution that would happen
    after applying privatiation noise

    :param distribution: The original distribution
    :param p: Privatization noise parameter
    :return: after-privatization-distribution

    Example:
    Let's start with a distribution concentrated in a single value:
    >>> distribution = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> privatize(distribution, 1.0)
    array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> # next, if we never tell truth with p=0.0 and true value is always the same, then we never emit this value
    >>> privatize(distribution, 0.0)
    array([0.11111111, 0.        , 0.11111111, 0.11111111, 0.11111111,
           0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111])
    >>> # next, uniform remains uniform
    >>> distribution = np.ones(10) * 0.1
    >>> privatize(distribution, 0.66)
    array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    >>> distribution = np.array([0.3, 0.4, 0.2, 0.1])
    >>> privatize(distribution, 0.8)
    array([0.28666667, 0.36      , 0.21333333, 0.14      ])
    """
    n = distribution.shape[0]
    assert np.isclose(distribution.sum(), 1.0)
    # if the metric took some value, then with probability p it remains the same
    # then, with probability (1.0 - p) it will go
    smoothed = distribution * p + (1.0 - p) * (1.0 - distribution) / (n - 1.0)
    assert np.isclose(
        smoothed.sum(), 1.0), f"sum({smoothed}) = {smoothed.sum()}"
    return smoothed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--list_length', default=5, type=int,
                        help="Depth of the ranking list used in the `calculate_epsilon` regime.")
    args = parser.parse_args()
    np.random.seed(args.seed)

    n = args.list_length + 1  # MaxRR has list_length + 1 values
    for p in [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]:
        print("Privatization noise parameter, p = ", p)
        for model in [PERFECT_MODEL, INFORMATIONAL_MODEL, NAVIGATIONAL_MODEL]:
            distributions = enumerate_distributions(
                args.list_length, model, p)
            print(
                f"\tUnder {model.name} ε = {get_most_dissimilar(distributions)[0]}")
        print("\tTheoretical bound of ε = ",
              np.log(p / (1.0 - p) * (n - 1)))
