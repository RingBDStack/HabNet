# HabNet
The code and data for our paper "Hierarchical Bi-Directional Self-Attention Networks for Paper Review Rating Recommendation" which is accepted by COLING 2020.

## Code overview
There are two folders of code: HabNet is the code for the main task of predicting final acceptance decisions for papers; HabNet_MC is the code for the sub_task of predicting ratings for reviews. The steps to run them are the same as shown in following section "Usage".

## Dataset
The OpenReview dataset (processed) collected by us are in the folder "OpenReview_Data" which includes two files "ICLR_Review_all_with_decision_processed.csv" and "ICLR_Review_all_processed.csv". There is also a copy of these two files in the "data" folder of HabNet and HabNet_MC respectively. "ICLR_Review_all_with_decision_processed.csv" is used for the main task of predicting acceptance decsions for papers, its copy is located in "data" folder of HabNet; "ICLR_Review_all_processed.csv" is used for the sub-task of predicting ratings for reviews, its copy is located in "data" folder of HabNet_MC.


## Code

## Requirements

- Python 3.6.8
- Tensorflow = 1.13.1
- Pandas
- Nltk
- Tqdm
- [Glove pre-trained word embeddings](http://nlp.stanford.edu/data/glove.6B.zip)

## Usage

First step: run script to prepare the data:

```bash
python data_prepare.py
```

Second step: train and evaluate the model:
<br>
*(make sure [Glove embeddings](#requirements) are ready before training, put glove.6B.50d.txt in HabNet, and put glove.6B.100d.txt in HabNet_MC)*
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```
```bash
python train.py
```

## Citation
If you use our code or dataset, please cite our COLING 2020 paper, our paper is available at: https://www.aclweb.org/anthology/2020.coling-main.555/

**Cite this paper using BibTex:**
@inproceedings{deng-etal-2020-hierarchical,
    title = "Hierarchical Bi-Directional Self-Attention Networks for Paper Review Rating Recommendation",
    author = "Deng, Zhongfen  and
      Peng, Hao  and
      Xia, Congying  and
      Li, Jianxin  and
      He, Lifang  and
      Yu, Philip",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.555",
    doi = "10.18653/v1/2020.coling-main.555",
    pages = "6302--6314",
}


## Acknowledgements
Our code is based on [DiSAN](https://github.com/taoshen58/DiSAN), we thank the authors of DiSAN for their open-source code.


