from pathlib import Path
import json
import sys
import numpy as np
from configuration.configparser import get_top_path

"""
Takes predictions from all axial planes and aggregates them together to a final
prediction. Prints the performance
"""


anatomical_planes = ["axial", "sagittal", "coronal"]

cfg_target = "generated"
cfgs_path = Path('./configuration')
data_path = Path('/home/helle246/data/sixkidneys/training')
get_path = Path('/home/helle246/data/urefine/pretrained_weights')
put_path = Path('/home/helle246/data/urefine/predictions')
if not put_path.exists():
    put_path.mkdir()

meta = json.load((cfgs_path / "meta" / sys.argv[1]).open())
num_training_sessions = len(meta["levels"])

def get_performance(predictions, truth):
    accuracy = np.sum(
        np.equal(truth, predictions).astype(np.float32)
    )/(100*512*512)
    true_positives = np.sum(
        ((truth + predictions) == 2).astype(np.int32)
    )
    false_positives = np.sum(
        np.logical_and(
            predictions == 1,
            truth == 0
        ).astype(np.int32)
    )
    false_negatives = np.sum(
        np.logical_and(
            truth == 1,
            predictions == 0
        ).astype(np.int32)
    )
    precision = true_positives/(true_positives+false_positives)
    recall = true_positives/(true_positives+false_negatives)
    f1 = 2*precision*recall/(precision + recall)

    return (
        accuracy,
        precision,
        recall,
        f1
    )





def get_probs(out, plane):
    parent = get_path / ("%s_%s_%d" % (out, plane, num_training_sessions))
    best = get_top_path(parent)
    return np.load(str(best/"probs.npy"))

for out in meta["outs"]:
    probs = get_probs(out, "axial")
    probs = probs + np.transpose(get_probs(out, "sagittal"), [2,1,0,3])
    probs = probs + np.transpose(get_probs(out, "coronal"), [1,0,2,3])

    predictions = (probs > 1.5).astype(np.int64)

    pred_file = put_path / ("%s.npy" % out)

    np.save(str(pred_file), predictions)

    truth = np.load(data_path / ('axial-%s-y.npy' % out))

    accuracy, precision, recall, f1 = get_performance(predictions, truth)

    print("******************************************************")
    print("ACCURACY:", accuracy)
    print("PRECISION:", precision)
    print("RECALL:", recall)
    print("F1:", f1)
    print("******************************************************")
