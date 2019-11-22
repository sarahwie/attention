import argparse
import time
import sys
parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str)
parser.add_argument('--encoder', type=str, choices=['lstm', 'average'], required=True)
parser.add_argument('--attention', type=str, choices=['tanh', 'frozen', 'pre-loaded'], required=False)
parser.add_argument('--n_iters', type=int, required=False, default=8)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--gold_label_dir', type=str, required=False)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--lmbda', type=float, required=False)
parser.add_argument('--adversarial', action='store_const', required=False, const=True)


args, extras = parser.parse_known_args()
args.extras = extras
args.command = ' '.join(['python'] + sys.argv)

from attention.Trainers.DatasetBC import datasets
from attention.ExperimentsBC import train_dataset_on_encoders

import torch
import numpy as np

# check that have provided a data directory to load attentions/predictions from
if (args.attention == 'pre-loaded' or args.adversarial) and not args.gold_label_dir :
    raise Exception("You must specify a gold-label directory for attention distributions") 

#check that have provided the correct dir:
if args.gold_label_dir and args.dataset.lower() not in args.gold_label_dir and args.dataset not in args.gold_label_dir :
    raise Exception("Gold-attention labels directory does not match specified dataset")

# add check for lmbda value if adversarial model
if args.adversarial and not args.lmbda :
    raise Exception("Must specify a lambda value for the adversarial model")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.adversarial :
    args.frozen_attn = False
    args.pre_loaded_attn = False
elif args.attention == 'frozen' :
    args.frozen_attn = True
    args.pre_loaded_attn = False
elif args.attention == 'tanh' :
    args.frozen_attn = False
    args.pre_loaded_attn = False
elif args.attention == 'pre-loaded': # not an adversarial model
    args.frozen_attn = False
    args.pre_loaded_attn = True
else :
    raise LookupError("Attention not found ...")

dataset = datasets[args.dataset](args)

if args.output_dir is not None :
    dataset.output_dir = args.output_dir

if args.adversarial :
    exp_name = '+'.join((args.encoder, 'adversarial'))
else :
    exp_name = '+'.join((args.encoder, args.attention))

start = time.time()
train_dataset_on_encoders(dataset, args, exp_name)
print("TOTAL ELAPSED TIME: %f HOURS OR %f MINUTES" % (((time.time() - start)/60/60), ((time.time() - start)/60)))




