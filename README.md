# Attention is not not Explanation

Code for the EMNLP 2019 paper *[Attention is not not Explanation](https://arxiv.org/abs/1908.04626)* by Wiegreffe & Pinter.

When using this codebase, please cite:
```
@inproceedings{wiegreffe2019attention,
  title={Attention is not not Explanation},
  author={Wiegreffe, Sarah and Pinter, Yuval},
  booktitle={"Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing"},
  year={2019},
  month = nov
  address = "Hong Kong"
  publisher = "Association for Computational Linguistics"
}
```

We've based our repository on the [code](https://github.com/successar/AttentionExplanation) provided by Sarthak Jain & Byron Wallace for their paper *[Attention is not Explanation](https://arxiv.org/abs/1902.10186)*.

Dependencies
--------------
Please refer to the installation instructions for the repository provided by [Jain & Wallace](https://github.com/successar/AttentionExplanation). We use the same dependencies.

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

Running Adversarial Model Experiments (Section 4)
--------------
- `./run_adversarial.sh [Diabetes, Anemia, AgNews, 20News_sports, imdb, sst] [lambda_value] [path/to/saved/model/with/gold/attentions/and/predictions]`

Remaining Todos
--------------
- 3.3 BOWs experiments
