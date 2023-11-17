# KG-LRR pytorch
This is the Pytorch implementation for our paper: Reinforced logical reasoning over KGs for interpretable recommendation system.

## Enviroment Requirement
numpy==1.21.0

pandas==1.3.0

scikit_learn==1.1.1

scipy==1.6.0

torch==1.8.1

torch_geometric==1.7.2

tqdm==4.61.2

## Dataset
You can get them from repository KGAT and PGPR, and use Freebase to create KG.
We will upload the processed data further.

## An Example to run KG-LRR
` cd code && python main.py --dataset=yelp2018 `
