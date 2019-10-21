import os
import pickle
import numpy as np
import json

def sortbylength(X, y) :
    len_t = np.argsort([len(x) for x in X])
    X1 = [X[i] for i in len_t]
    y1 = [y[i] for i in len_t]
    return X1, y1
    
def filterbylength(X, y, min_length = None, max_length = None) :
    lens = [len(x)-2 for x in X]
    min_l = min(lens) if min_length is None else min_length
    max_l = max(lens) if max_length is None else max_length

    idx = [i for i in range(len(X)) if len(X[i]) > min_l+2 and len(X[i]) < max_l+2]
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    return X, y

def set_balanced_pos_weight(dataset) :
    y = np.array(dataset.train_data.y)
    dataset.pos_weight = [len(y) / sum(y) - 1]

class DataHolder() :
    def __init__(self, X, y, y_attn=None, true_pred=None) :
        self.X = X
        self.y = y
        self.gold_attns = y_attn
        self.true_pred = true_pred
        self.attributes = ['X', 'y', 'gold_attns', 'true_pred']


class Dataset() :
    def __init__(self, name, path, min_length=None, max_length=None, args=None) :
        self.name = name
        if args is not None and hasattr(args, 'data_dir') :
            path = os.path.join(args.data_dir, path)
            
        self.vec = pickle.load(open(path, 'rb'))

        X, Xt = self.vec.seq_text['train'], self.vec.seq_text['test'] # these are lists (of lists) of num. insts-length (NOT PADDED)
        y, yt = self.vec.label['train'], self.vec.label['test']

        X, y = filterbylength(X, y, min_length=min_length, max_length=max_length)
        Xt, yt = filterbylength(Xt, yt, min_length=min_length, max_length=max_length)
        Xt, yt = sortbylength(Xt, yt)

        if args.pre_loaded_attn or args.adversarial :
            # these are lists of lists, with some residual padding
            y_attn = json.load(open(os.path.join(args.gold_label_dir, 'train_attentions_best_epoch.json'), 'r'))
            yt_attn = json.load(open(os.path.join(args.gold_label_dir, 'test_attentions_best_epoch.json'), 'r'))

            true_pred = json.load(open(os.path.join(args.gold_label_dir, 'train_predictions_best_epoch.json'), 'r'))
            true_pred_t = json.load(open(os.path.join(args.gold_label_dir, 'test_predictions_best_epoch.json'), 'r'))
            true_pred = [e[0] for e in true_pred]
            true_pred_t = [e[0] for e in true_pred_t] #these are lists of num. insts-length

            #trim padding from static attentions
            new_attns = []
            for e, a in zip(X, y_attn):
                tmp = [0] + [el for el in a if el != 0] + [0]
                assert len(tmp) == len(e)
                new_attns.append(tmp)
            y_attn = new_attns

            #do the same for test
            new_attns = []
            for e, a in zip(Xt, yt_attn):
                tmp = [0] + [el for el in a if el != 0] + [0]
                assert len(tmp) == len(e)
                new_attns.append(tmp)
            yt_attn = new_attns

            self.train_data = DataHolder(X, y, y_attn, true_pred)
            self.test_data = DataHolder(Xt, yt, yt_attn, true_pred_t)

        else :
            self.train_data = DataHolder(X, y)
            self.test_data = DataHolder(Xt, yt)

        if args is not None and hasattr(args, 'hidden_size') :
            self.hidden_size = args.hidden_size
        
        self.output_size = 1
        self.save_on_metric = 'roc_auc'
        self.keys_to_use = {
            'roc_auc' : 'roc_auc', 
            'pr_auc' : 'pr_auc'
        }

        self.bsize = 32
        if args is not None and hasattr(args, 'output_dir') :
            self.basepath = args.output_dir
            

########################################## Dataset Loaders ################################################################################

def SST_dataset(args=None) :
    dataset = Dataset(name='sst', path='preprocess/SST/vec_sst.p', min_length=5, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def IMDB_dataset(args=None) :
    dataset = Dataset(name='imdb', path='preprocess/IMDB/vec_imdb.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def News20_dataset(args=None) :
    dataset = Dataset(name='20News_sports', path='preprocess/20News/vec_20news_sports.p', min_length=6, max_length=500, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def ADR_dataset(args=None) :
    dataset = Dataset(name='tweet', path='preprocess/Tweets/vec_adr.p', min_length=5, max_length=100, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def Anemia_dataset(args=None) :
    dataset = Dataset(name='anemia', path='preprocess/MIMIC/vec_anemia.p', max_length=4000, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def Diabetes_dataset(args=None) :
    dataset = Dataset(name='diabetes', path='preprocess/MIMIC/vec_diabetes.p', min_length=6, max_length=4000, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def AGNews_dataset(args=None) :
    dataset = Dataset(name='agnews', path='preprocess/ag_news/vec_agnews.p', args=args)
    set_balanced_pos_weight(dataset)
    return dataset

datasets = {
    "sst" : SST_dataset,
    "imdb" : IMDB_dataset,
    "20News_sports" : News20_dataset,
    "tweet" : ADR_dataset ,
    "Anemia" : Anemia_dataset,
    "Diabetes" : Diabetes_dataset,
    "AgNews" : AGNews_dataset
}
