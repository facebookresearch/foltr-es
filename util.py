# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np


def smoothen_trajectory(ys, group_size=2):
    """
    Applies sliding average over the window of size `group_size`;
    no padding is applied on the edges.

    >>> smoothen_trajectory(np.array([0.0, 1.0, 1.0, 0.0]))
    array([0.5, 1. , 0.5])
    >>> smoothen_trajectory(np.array([0.0, 1.0, 1.0, 0.0]), group_size=3)
    array([0.66666667, 0.66666667])
    """
    return np.convolve(np.ones(group_size)/group_size, ys, "valid")
