# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import torch
from sklearn import linear_model
from foltr.client.rankers import LinearRanker
from foltr.client.click_simulate import NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL, PERFECT_MODEL
from foltr.client.metrics import ExpectedMaxRR
from foltr.data.datasets import MqDataset, DataSplit


def lsq_ranker(qrels: DataSplit) -> torch.nn.Module:
    """
    Trains a least-square regression ranker on labels
    :param qrels: query -> QueryDoc dict
    :return: Trained ranker
    """
    y = np.hstack(x.relevance_labels for x in qrels.values())
    x = np.vstack(x.features for x in qrels.values())
    # fit with intercept then ignore it
    reg = linear_model.Ridge(alpha=0.0, fit_intercept=True)
    reg.fit(x, y)
    coefs = reg.coef_.reshape(1, reg.coef_.shape[0])
    model = LinearRanker(x.shape[1])
    model.fc.weight.data = torch.from_numpy(coefs).float()
    return model


def get_svm_ranker(fold_id: int, dataset_name: str, num_features: int) -> torch.nn.Module:
    """
    Spawns a process running SVM Rank, reads the trained model from the file. For that, uses
    the dataset directly. Requires that the svmrank binary is in the ./svmrank dir

    :param fold_id: Fold's number
    :param dataset_name: Name of the dataset (ie mq2007 or mq2008)
    :param num_features: Number of features to be expected.
    :return: Trained linear model
    """
    import tempfile
    import subprocess
    import os

    model_file = tempfile.mkstemp(dir="./cache/")[1]
    sorted_by_reqid = f"data/{dataset_name}/Fold{fold_id+1}/train.reqid.txt"
    if not os.path.exists(sorted_by_reqid):
        raw_train = f"data/{dataset_name}/Fold{fold_id+1}/train.txt"
        print(f"Converting from {raw_train} to {sorted_by_reqid}")
        convert2svmrank(raw_train, sorted_by_reqid)

    subprocess.check_output(
        ["./svmrank/svm_rank_learn", "-c", "1000.0", sorted_by_reqid, model_file])
    coefs = np.zeros(num_features, dtype=np.float32)
    with open(model_file, 'r') as f:
        for coeff_pair in f.readlines()[-1].split()[1:-1]:
            index, value = coeff_pair.split(":")
            index = int(index) - 1  # it is 1-based in the file
            value = float(value)
            coefs[index] = value
    coefs = coefs.reshape(1, num_features)
    model = LinearRanker(num_features)
    model.fc.weight.data = torch.tensor(coefs).float()
    return model


def convert2svmrank(path_in, path_out):
    with open(path_in, 'r') as inp, open(path_out, 'w') as outp:
        inmem = inp.readlines()
        keys = [int(l.split()[1].split(":")[1]) for l in inmem]
        inmem = sorted(list(zip(keys, inmem)))
        for _, line in inmem:
            outp.write(line)


if __name__ == '__main__':
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)
    click_models = [NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL, PERFECT_MODEL]

    dataset_names = ["MQ2007", "MQ2008"]
    datasets = [MqDataset.from_path("./data/MQ2007/", "./cache/"),
                MqDataset.from_path("./data/MQ2008/", "./cache/")]

    # mapping: baseline -> dataset -> {train, test} -> click model -> fold -> performance storage
    def cm_hash(): return dict((cm.name, []) for cm in click_models)

    def dataset_hash(): return dict(
        (name, {'train': cm_hash(), 'test': cm_hash()}) for name in dataset_names)
    results = {'lsq': dataset_hash(), 'svmrank': dataset_hash()}

    for dataset_name, dataset in zip(dataset_names, datasets):
        for click_model in click_models:
            metric = ExpectedMaxRR(click_model)
            for fold_id, fold in enumerate(dataset.folds):
                lsq_train = metric.eval_ranker(
                    lsq_ranker(fold.train), fold.train)
                lsq_test = metric.eval_ranker(
                    lsq_ranker(fold.train), fold.test)
                results['lsq'][dataset_name]['train'][click_model.name].append(
                    lsq_train)
                results['lsq'][dataset_name]['test'][click_model.name].append(
                    lsq_test)

                svmrank_train = metric.eval_ranker(
                    get_svm_ranker(fold_id, dataset_name, 46), fold.train)
                svmrank_test = metric.eval_ranker(
                    get_svm_ranker(fold_id, dataset_name, 46), fold.test)
                results['svmrank'][dataset_name]['train'][click_model.name].append(
                    svmrank_train)
                results['svmrank'][dataset_name]['test'][click_model.name].append(
                    svmrank_test)

                print(f"Expected MaxRR on {dataset_name}, under {click_model.name}, fold={fold_id}"
                      f"\tLSQ: Train {lsq_train}, Test {lsq_test}",
                      f"\tSVMRank: Train {svmrank_train}, Test {svmrank_test}")

    with open('baselines.json', 'w') as fout:
        fout.write(json.dumps(results).lower())
