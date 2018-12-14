# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict
import numpy as np


class CcmClickModel:
    """
    An implementation of the CCM click model, see the paper for the details and references.
    """

    def __init__(self, click_relevance: Dict[int, float], stop_relevance: Dict[int, float], name: str, depth: int):
        """
        :param click_relevance: A mapping from a relevance label to the probability of the doc being clicked (assuming
        it is examined)
        :param stop_relevance: A mapping from a relevance label to the probability of the user stopping after clicking
        on this document
        :param name: Name of the click model instance
        :param depth: How deep the user examines the result page
        """
        self.click_relevance = click_relevance
        self.stop_relevance = stop_relevance
        self.name = name
        self.depth = depth

    def __call__(self, relevances: np.ndarray, random_state: np.random.RandomState) -> np.ndarray:
        """
        Generates an indicator array of the click events for the ranked documents with relevance labels encoded in
        `relevances`.
        :param relevances: Relevance labels of the documents
        :param random_state: Random generator state
        :return: Indicator array of the clicks on the documents

        As an example, consider a model of a user who always clicks on a highly relevant result and immediately stops
        >>> model = CcmClickModel(click_relevance={0: 0.0, 1: 0.0, 2: 1.0},
        ...                       stop_relevance={0: 0.0, 1: 0.0, 2: 1.0}, name="Model", depth=10)
        >>> # With the result list with highly relevant docs on positions 2 and 4,
        >>> doc_relevances = np.array([1, 0, 2, 0, 2, 0])
        >>> # We expect the user to click on the 3rd document, as it the first highly relevant:
        >>> model(doc_relevances, np.random.RandomState(1)).tolist()
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        """
        n_docs = relevances.shape[0]
        result = np.zeros(n_docs)
        for i in range(min(self.depth, n_docs)):
            r = relevances[i]
            p_click = self.click_relevance[r]
            p_stop = self.stop_relevance[r]

            if random_state.uniform() < p_click:
                result[i] = 1
            if result[i] == 1 and random_state.uniform() < p_stop:
                break
        return result


# Those are three standard models used in the literature
PERFECT_MODEL = CcmClickModel(click_relevance={0: 0.0, 1: 0.5, 2: 1.0},
                              stop_relevance={0: 0.0, 1: 0.0, 2: 0.0}, name="Perfect", depth=10)
NAVIGATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.05, 1: 0.5, 2: 0.95},
                                   stop_relevance={0: 0.2, 1: 0.5, 2: 0.9}, name="Navigational", depth=10)
INFORMATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.4, 1: 0.7, 2: 0.9},
                                    stop_relevance={0: 0.1, 1: 0.3, 2: 0.5}, name="Informational", depth=10)
