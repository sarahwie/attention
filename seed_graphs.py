#!/usr/bin/env python
# coding: utf-8
import argparse
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from glob import glob

from attention.common_code.common import jsd, get_latest_model
from attention.common_code.plotting import annotate, \
            plot_violin_by_class, plot_scatter_by_class, \
            init_gridspec, adjust_gridspec, show_gridspec, save_axis_in_file
from attention.Trainers.DatasetBC import datasets



### TODO parametrize, or just take from data_full_dir arg
ADVERSARY_MODE = True


def numpify(yh):
    return np.array([np.array(att[1:-1]) for att in yh])

    
def main():
    parser = argparse.ArgumentParser(description='Plot seed distribution differences for a dataset')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='lstm+tanh')
    parser.add_argument('--data_dir', type=str, default='.')
    if ADVERSARY_MODE:
        parser.add_argument('--data_full_dir', type=str, default='.')
    args = parser.parse_args()
    
    plt.switch_backend('agg')
    
    # load data
    data_name = args.dataset
    model_type = args.model_type
    m, t = model_type.split('+')
    dataset = datasets[data_name]({'encoder':f'm', 'attention':f't','data_dir':args.data_dir})
    test_data = dataset.test_data
    
    # load 'original' model
    if ADVERSARY_MODE:
        base_dir = args.data_full_dir
    else:
        base_dir = get_latest_model(f'outputs/{data_name.lower()}/{model_type}/')
    
    yhat_orig = json.load(open(os.path.join(base_dir, 'test_attentions_best_epoch.json')))
    yhat_preds_orig = json.load(open(os.path.join(base_dir, 'test_predictions_best_epoch.json')))

    # load seed models
    yhat_seeds = []
    yhat_preds_seeds = []
    
    
    if ADVERSARY_MODE:
        for i in range(2):
            yhat_seeds.append(json.load(open(f'{base_dir}/test_attentions_jw_adversary_{i:02d}.json')))
            yhat_preds_seeds.append(json.load(open(f'{base_dir}/test_predictions_jw_adversary_{i:02d}.json')))
    else:
        for sd in glob(f'outputs/seed_*/{data_name.lower()}/{model_type}/*'):
            yhat_seeds.append(json.load(open(os.path.join(sd, 'test_attentions_best_epoch.json'))))
            yhat_preds_seeds.append(json.load(open(os.path.join(sd, 'test_predictions_best_epoch.json'))))

    yhat_orig = numpify(yhat_orig)
    yhat_seeds = [numpify(sd) for sd in yhat_seeds]

    # compute attention diffs, prediction diffs
    num_rands = len(yhat_seeds)
    seed_atts = np.zeros((num_rands, len(test_data.y)))
    max_attn = np.zeros((num_rands, len(test_data.y)))
    for i, yhs in enumerate(yhat_seeds):
        for j in range(len(test_data.y)):
            seed_atts[i,j] = jsd(yhat_orig[j], yhs[j])
            max_attn[i,j] = max(yhat_orig[j])
    max_attn = np.median(max_attn, 0)

    attn_diff = np.median(seed_atts, 0)

    y_diff = np.abs(np.array(yhat_preds_seeds) - np.array(yhat_preds_orig)).mean(0).flatten()

    # output
    if ADVERSARY_MODE:
        dirname = 'seed_graphs_adv'
    else:
        dirname = 'seed_graphs'

    fig, axes = init_gridspec(1,2,2)
    
    plot_violin_by_class(axes[0], max_attn, attn_diff, np.array(yhat_preds_orig), xlim=(0, 1.0))
    if ADVERSARY_MODE:
        annotate(axes[0], xlim=(-0.05, 0.7), ylabel="Max Attention", xlabel="JSD (adversary vs base)", legend=None)
    else:
        annotate(axes[0], xlim=(-0.05, 0.7), ylabel="Max Attention", xlabel="JSD (seeds vs base)", legend=None)
    
    plot_scatter_by_class(axes[1], attn_diff, y_diff, np.array(yhat_preds_orig))
    if ADVERSARY_MODE:
        annotate(axes[1], xlim=(-0.05, 0.7), ylim=(-0.05, 1.05), xlabel="JSD (adversary vs base)", ylabel='Output Difference', legend=None)
    else:
        annotate(axes[1], xlim=(-0.05, 0.7), ylim=(-0.05, 1.05), xlabel="JSD (seeds vs base)", ylabel='Output Difference', legend=None)
    
    adjust_gridspec()
    if ADVERSARY_MODE:
        save_axis_in_file(fig, axes[0], dirname, f'{data_name}-{model_type}-adv-diffs')
        save_axis_in_file(fig, axes[1], dirname, f'{data_name}-{model_type}-adv-scatter')
    else:
        save_axis_in_file(fig, axes[0], dirname, f'{data_name}-{model_type}-seed-diffs')
        save_axis_in_file(fig, axes[1], dirname, f'{data_name}-{model_type}-seed-scatter')
    show_gridspec()
    print('Complete')
    
    
if __name__ == '__main__':
    main()


