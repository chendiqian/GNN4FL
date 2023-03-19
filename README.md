# FedGNN: our paper title

## Environment setup

### install pytorch, see [PyTorch](https://pytorch.org/get-started/previous-versions/)
`conda install pytorch torchvision -c pytorch`

### PyTorch Geometric, see [PyG](https://pytorch-geometric.readthedocs.io/en/latest/#)
`pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html`

`pip install jupyterlab`

## CNN models dataset creation
The project is not aimed at training a CNN with SOTA result, but to analyze the performance of federated learning. 
Therefore, we train a CNN with a simple architecture. The example script is e.g.

`python create_cnn_models.py --digitsPermodel 10`

which will give a set of models trained on different digits with different initialization seeds. The trained models will be used as a dataset.

For FEMNIST dataset, we use the [LEAF](https://github.com/TalwalkarLab/leaf) repo, please see to the official repo and create FEMNIST dataset. 
After that, simply copy the `train` and `test` folder under `./datasets/FEMNIST`.

## Graph dataset creation
For each graph, we have several models as parameter nodes, and several virtual aggregator nodes. Run e.g.

`python create_graph_datasets.py` with desired hyperparameters. Remember to specify `--cnn_db` for the CNN dataset.

## Train GNN and MLP baseline
Once you have created the graph datasets, you can train a heterogeneous GNN on them.
Simply run `train_gnn.py` or `mlp_baseline.py` with deep learning hyperparameters.
