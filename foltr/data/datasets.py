# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, NamedTuple, Optional, Dict
import numpy as np
import os
import hashlib
import pickle

# all docs associated to a particular query
QueryDocs = NamedTuple('QueryDocs', [
    ('relevance_labels', np.ndarray),  # relevance labels, 1D array
    ('features', np.ndarray)])  # document-query features, 2D array (# docs x # features)

# each dataset split is a mapping between a query id to the features/relevance arrays
DataSplit = Dict[str, QueryDocs]
FoldRecord = NamedTuple("FoldRecord", [("train", DataSplit),
                                       ("validation", DataSplit),
                                       ("test", DataSplit)])


class MqDataset:
    """
    Storage for a dataset; essentially a list of FolRecords with a name.
    For usage, check e.g. tests/test_dataset.py
    """
    # In both MQ2007 & MQ2008, we have 46 features
    n_features = 46

    def __init__(self, name: str, folds: List[FoldRecord]):
        self.folds = folds
        self.name = name

    @classmethod
    def from_path(cls, root_path: str, cache_root: Optional[str]) -> 'MqDataset':
        """
        Constructs a dataset by reading form a disk
        :param root_path: A path to the root that contains (Fold1, Fold2, ...)
        :param cache_root: None if no caching is needed, otherwise a path to the cache dir;
                        if the cache dir already contains the data it will be used. Hence, cleaning
                        of the cache has to be done manually if needed.
        :return: Constructed Dataset instance
        """
        return MqDataset(root_path, MqDataset.read_mq(root_path, MqDataset.n_features, cache_root))

    @staticmethod
    def read_mq_file(path: str, n_features: int, cache_root: Optional[str]) -> DataSplit:
        """
        Reads a single MQ file, such as Fold1/train.txt
        :param path: Path to the file
        :param n_features: Number of features to be expected
        :param cache_root: Optional root dir for the cache; no caching if None
        :return: Filled in DataSplit instance
        """
        if cache_root is not None:
            cache_name = f"{hashlib.md5(path.encode()).hexdigest()}.pkl"
            file_path = f'./{cache_root}/{cache_name}'
            if os.path.exists(file_path):
                print(f"Loading from cache file {file_path}")
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                return data

        query2data = {}

        with open(path, 'r') as f:
            for line in f:
                features = [0.0 for _ in range(n_features)]
                tail = line.find("#")
                line = line[:tail].split()
                relevance = int(line[0])
                qid = line[1]
                if qid not in query2data:
                    query2data[qid] = [[], []]

                for x in line[2:]:
                    x = x.split(":")
                    # one-based
                    index, value = int(x[0]) - 1, float(x[1])
                    features[index] = value

                query2data[qid][0].append(relevance)
                query2data[qid][1].append(features)
        for qid, query_data in query2data.items():
            relevance_labels, features = query_data
            query2data[qid] = QueryDocs(relevance_labels=np.array(relevance_labels),
                                        features=np.array(features))

        if cache_root is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(query2data, f)
            print(f"Cached the array to {file_path}")
        return query2data

    @staticmethod
    def read_mq(root_path: str, n_features: int, cache_root: Optional[str]) -> List[FoldRecord]:
        """
        Reads all 5 folds of a MQ200{7,8} dataset
        :param root_path: Dataset root that contains (Fold1, Fold2, ...)
        :param n_features: Number of features to be expected
        :param cache_root: Optional root dir for the cache; no caching if None
        :return: Per-fold FoldRecord
        """
        folds = []
        for fold_dir in [f"{root_path}/Fold{i}" for i in range(1, 6)]:
            train = MqDataset.read_mq_file(
                f"{fold_dir}/train.txt", n_features, cache_root)
            test = MqDataset.read_mq_file(
                f"{fold_dir}/test.txt", n_features, cache_root)
            validation = MqDataset.read_mq_file(
                f"{fold_dir}/vali.txt", n_features, cache_root)
            folds.append(FoldRecord(
                train=train, test=test, validation=validation))
        return folds
