# Attention is not not Explanation

Code for the EMNLP 2019 paper *[Attention is not not Explanation](https://www.aclweb.org/anthology/D19-1002/)* by Wiegreffe & Pinter.

When using this codebase, please cite:
```
@inproceedings{wiegreffe-pinter-2019-attention,
    title = "Attention is not not Explanation",
    author = "Wiegreffe, Sarah  and
      Pinter, Yuval",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1002",
    doi = "10.18653/v1/D19-1002",
    pages = "11--20"
}
```

We've based our repository on the [code](https://github.com/successar/AttentionExplanation) provided by Sarthak Jain & Byron Wallace for their paper *[Attention is not Explanation](https://arxiv.org/abs/1902.10186)*.

Dependencies
--------------
Please refer to the installation instructions for the repository provided by [Jain & Wallace](https://github.com/successar/AttentionExplanation). We use the same dependencies.
Also, make sure to export the meta-directory into which you clone `attention` to your PYTHONPATH in order for the imports to work correctly. For example, if the path to the cloned directory is `/home/users/attention/`, then run `export PYTHONPATH='/home/users'`.

Data Preprocessing
--------------
Please perform the preprocessing instructions provided by Jain & Wallace [here](https://github.com/successar/AttentionExplanation/tree/master/preprocess). We replicated these instructions for the `Diabetes`, `Anemia`, `SST`, `IMDb`, `AgNews`, and `20News` datasets.

Running Baselines
--------------
We replicate the reported baselines in Jain & Wallace's paper (as reported in our paper in Table 2) by running the following commands:
- `./run_baselines.sh [Diabetes, Anemia, AgNews, 20News_sports, imdb, sst]`

Freezing the Attention Distribution (Section 3.1)
--------------
- `./run_frozen_attention.sh [Diabetes, Anemia, AgNews, 20News_sports, imdb, sst]`

Running Random Seeds Experiments (Section 3.2)
--------------
- `parallel ./run_seeds.sh :::: seeds.txt ::: sst AgNews imdb 20News_sports Diabetes Anemia`
- Code for constructing the violin plots in Figure 3 can be found in `seed_graphs.py` and `Seed_graphs.ipynb`.

Running BOWs Experiments (Section 3.3)
--------------
- To run the Bag of Words model with trained (MLP) attention weights: `./run_bows_baselines.sh [Diabetes, Anemia, AgNews, 20News_sports, imdb, sst]`
- To run the Bag of Words model with uniform attention weights: `./run_bows_frozen_attn.sh [Diabetes, Anemia, AgNews, 20News_sports, imdb, sst]`
- To run the Bag of Words model with frozen attention weights from another model: `./run_bows_set_to_pretrained_distribution.sh [Diabetes, Anemia, AgNews, 20News_sports, imdb, sst] [path/to/saved/model/with/attention/weights]`

Running Adversarial Model Experiments (Section 4)
--------------
- `./run_adversarial.sh [Diabetes, Anemia, AgNews, 20News_sports, imdb, sst] [lambda_value] [path/to/saved/model/with/gold/attentions/and/predictions]`
