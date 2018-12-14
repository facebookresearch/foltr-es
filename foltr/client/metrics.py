# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import NamedTuple, Union, List
import numpy as np
import random
import torch
from foltr.data import DataSplit
from .click_simulate import CcmClickModel


# we need both privatized and not: the first is actually used for the optimization, the second is used
# to get the figures
MetricValue = NamedTuple(
    'MetricValue', [('privatized', float), ('non_privatized', float)])


class PrivatizedMaxRR:
    """
    Implements a privatized MaxRR metric with a privatization noise.
    """

    def __init__(self, p: float, cutoff: int = 10):
        """
        :param p: Privatization parameter, within [0, 1]. Essentially, the probability of the output not being corrupted
         - with p = 1.0 returns true MaxRR all the time.
        :param cutoff: Max depth of the ranked lists. The returned metric values take cutoff + 1 values.
        """
        self.p = p
        self.cutoff = cutoff
        # zero for the case when no click occurred
        self.possible_outputs = [0.0] + \
            [1.0 / (1.0 + r) for r in range(cutoff)]

    def __call__(self, clicks: Union[np.ndarray, List[float]]) -> MetricValue:
        """
        Calculates the metric value for the one-encoded clicks metrics.
        :param clicks: A numpy array encoding per-position click events with {0, 1}, e.g.
        [1, 0, 0, 0, 1] encodes clicks on positions 0 and 4 on a ranked list of length 5.
        :return: A MetricValue instance with privatized and true value metrics.
        Examples:
        >>> PrivatizedMaxRR(1.0)([1.0, 0.0, 0.0, 0.0])
        MetricValue(privatized=1.0, non_privatized=1.0)
        >>> PrivatizedMaxRR(1.0)([0.0, 0.0, 0.0, 0.0])
        MetricValue(privatized=0.0, non_privatized=0.0)
        """
        reciprocal_rank = 0.0
        n_docs = len(clicks)
        for i in range(min(self.cutoff, n_docs)):
            if clicks[i] > 0:
                reciprocal_rank = 1.0 / (1.0 + i)
                break
        if np.random.random() < self.p:
            # tell the true metric value
            return MetricValue(privatized=reciprocal_rank, non_privatized=reciprocal_rank)
        else:
            # reject sample until we get a different metric value
            while True:
                sampled_value = random.sample(self.possible_outputs, 1)[0]
                if sampled_value != reciprocal_rank:
                    return MetricValue(privatized=sampled_value, non_privatized=reciprocal_rank)


class ExpectedMaxRR:
    """
    This class is used to calculate the expectation of MaxRR over a ranked list given its relevance scores
    and a CCM click model. Essentially, it induces an offline metric.
    """

    def __init__(self, click_model: CcmClickModel, cutoff: int = 10):
        """
        :param click_model: An instance of CCM model; the expectation is calculated w.r.t. this model
        :param cutoff: The cut-off level - documents below it are not used to calculate the MaxRR
        """
        self.click_model = click_model
        self.cutoff = cutoff

    def eval_ranking(self, ranking: np.ndarray) -> float:
        """
        Maps a ranked list  into the expectation of MaxRR
        :param ranking: a np.ndarray vector of document relevance labels
        :return: the expected MaxRR w.r.t. click_model

        As an example, consider a model of a user who always clicks on a highly relevant result and immediately stops;
        under such a model MaxRR would be the reciprocal rank of the first highly relevant document:
        >>> model = CcmClickModel(click_relevance={0: 0.0, 1: 0.0, 2: 1.0},
        ...                       stop_relevance={0: 0.0, 1: 0.0, 2: 1.0}, name="Model", depth=10)
        >>> metric = ExpectedMaxRR(model)
        >>> doc_relevances = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 1.0])
        >>> metric.eval_ranking(doc_relevances)
        0.25
        >>> doc_relevances = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        >>> metric.eval_ranking(doc_relevances)
        0.0
        """
        click_relevance = self.click_model.click_relevance

        metric = 0.0
        p_not_clicked_yet = 1.0
        for i in range(min(self.cutoff, ranking.shape[0])):
            r = ranking[i]
            p_click = click_relevance[r]

            p_first_click = p_click * p_not_clicked_yet
            p_not_clicked_yet *= 1.0 - p_click
            metric += p_first_click / (i + 1.0)
        return metric

    def eval_ranker(self, ranker: torch.nn.Module, data: DataSplit) -> float:
        """
        Evaluates a ranker over all queries in a dataset provided in `data`. To do that, applies the ranker for a
        queries, sort the documents according to ranking scores, gets the quality of the score.
        :param ranker: A ranker to be evaluated.
        :param data: A dataset to use in the evaluation.
        :return: Averaged over all queries value of the ranking quality, assessed by ExpactedMaxRR.
        """

        average_metric = 0.0

        for slice in data.values():
            features = torch.from_numpy(slice.features).float()
            scores_slice = ranker.forward(features).data.numpy()[:, 0]
            ranking_order = np.argsort(scores_slice)[::-1]
            relevances = slice.relevance_labels[ranking_order]
            average_metric += self.eval_ranking(relevances)
        return average_metric / len(data)
