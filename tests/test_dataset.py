# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.resolve())
from foltr.data import MqDataset


def test_reading_file() -> None:
    """
    Ensure that a tiny file is read correctly
    """
    data_split = MqDataset.read_mq_file(
        './tests/tiny_data.txt', n_features=46, cache_root=None)
    # queries are re-numerated sequentially from zero
    assert set(data_split.keys()) == {'qid:1', 'qid:3'}
    assert data_split['qid:1'].features.shape == (1, 46)
    assert data_split['qid:3'].features.shape == (2, 46)

    # features are transformed to zero-based
    assert data_split['qid:1'].features[0, 0] == 0.0
    assert data_split['qid:3'].features[0, 0] == 2.0
    assert data_split['qid:3'].features[1, 4] == 3.0

    assert data_split['qid:1'].relevance_labels.tolist() == [0]
    assert data_split['qid:3'].relevance_labels.tolist() == [0, 2]


def test_reading_MQ2008():
    """
    Mostly double-checking that reading doesn't crash; not much more
    """
    mq2008 = MqDataset.from_path("./data/MQ2008/", cache_root=None)
    assert len(mq2008.folds) == 5

    for fold in mq2008.folds:
        train_queries, test_queries, validation_queries = [
            set(x) for x in [fold.train, fold.test, fold.validation]]

        assert len(test_queries) < len(train_queries)
        assert len(test_queries.intersection(train_queries)) == 0
        assert len(validation_queries) < len(train_queries)
        assert len(validation_queries.intersection(train_queries)) == 0

