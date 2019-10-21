from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score
import numpy as np
from pandas.io.json.normalize import nested_to_record
from collections import defaultdict
import pandas as pd
import torch
from IPython.display import display

def tvd(predictions, targets): #accepts two numpy arrays of dimension: (num. instances, )
    return (0.5 * np.abs(predictions - targets)).sum()

def batch_tvd(predictions, targets): #accepts two Torch tensors... " "
    return (0.5 * torch.abs(predictions - targets)).sum()

def calc_metrics_classification(target, predictions, target_scores=None, jsd_score=None) :

    if target_scores is not None :
        assert predictions.squeeze(1).shape == target_scores.shape
        tvdist = tvd(predictions.squeeze(1), target_scores)

    if predictions.shape[-1] == 1 :
        predictions = predictions[:, 0]
        predictions = np.array([1 - predictions, predictions]).T

    predict_classes = np.argmax(predictions, axis=-1)

    if len(np.unique(target)) < 4 :
        rep = nested_to_record(classification_report(target, predict_classes, output_dict=True), sep='/')
    else :
        rep = {}
    rep.update({'accuracy' : accuracy_score(target, predict_classes)})

    if jsd_score :
        rep.update({'js_divergence' : jsd_score})
    if target_scores is not None :
        rep.update({'TVD' : tvdist})

    if predictions.shape[-1] == 2 :
        rep.update({'roc_auc' : roc_auc_score(target, predictions[:, 1])})
        rep.update({"pr_auc" : average_precision_score(target, predictions[:, 1])})
    return rep

def print_metrics(metrics, adv=False) :
    tabular = {k:v for k, v in metrics.items() if '/' in k}
    non_tabular = {k:v for k, v in metrics.items() if '/' not in k}
    print(non_tabular)

    d = defaultdict(dict)
    for k, v in tabular.items() :
        if not k.startswith('label_') :
            d[k.split('/', 1)[0]][k.split('/', 1)[1]] = v
        if '/1/' in k or 'auc' in k:
            d[k.split('/', 1)[0]][k.split('/', 1)[1]] = v

    df = pd.DataFrame(d)
    with pd.option_context('display.max_columns', 30):
        display(df.round(3))

    if adv :
        print("TVD:", metrics['TVD'])
        print("JS:", metrics['js_divergence'])
        
