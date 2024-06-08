# Reinforced logical reasoning over KGs for interpretable recommendation system.

## Overview

This work is currently under review, and more complete dataset will be updated upon acceptance.

In this paper, we propose an ”encoder-decoder" architecture that uses a better encoding method for better logic decoding. The encoding module utilizes the KG and relation-aware attention mechanism to obtain robust embeddings. The decoding module models the human-like propositional logical system, mimics the "AND/OR/NOT" functions and generate the prediction and explanations.

## Main Environment Requirements

```
numpy==1.21.0

pandas==1.3.0

scikit_learn==1.1.1

scipy==1.6.0

torch==1.8.1

torch_geometric==1.7.2

tqdm==4.61.2
```

## Dataset

- You can find the full version of recommendation datasets via [Amazon-book](http://jmcauley.ucsd.edu/data/amazon), [Amazon-Electronics](http://jmcauley.ucsd.edu/data/amazon/), and [Yelp2018](https://www.yelp.com/dataset/challenge).
- We follow [KB4Rec](https://github.com/RUCDM/KB4Rec) to preprocess Amazon-book and Amazon-electronics datasets, mapping items into Freebase entities via title matching if there is a mapping available.
- We have updated processed data in the './data' folder. Due to the limit of file size, we cannot directly upload the original knowledge graphs. You can use those codes in [KB4Rec](https://github.com/RUCDM/KB4Rec) to transfer some other recommendation dataset. 

## Training & Testing

```
cd ./code
python main.py --dataset=yelp2018 --explain=True --test_start_epoch=10
```

The 'explain' parameter controls the decoding part whether outputs the explanations or not and the 'test_start_epoch' controls the start epoch to do test. In our original settings, KG-LRR will do test and output its explanations every 5 epochs.

Except from the above parameters, there are many other control parameters that can be detailed in 'main.py'.  Please pay attention to all the default settings and modify them as you want.