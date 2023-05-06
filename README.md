# Advancing Federated Learning in 6G: A Trusted Architecture with Graph-based Analysis

## Environment setup

### install pytorch, see [PyTorch](https://pytorch.org/get-started/previous-versions/)
`conda install pytorch torchvision -c pytorch`

### PyTorch Geometric, see [PyG](https://pytorch-geometric.readthedocs.io/en/latest/#)
`pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html`


### Encryption

See [paillier](https://github.com/data61/python-paillier)

`pip install phe`

### Other packages

`pip install tensorboard`
`pip install jupyterlab`

## Graph dataset creation
The project is not aimed at training a CNN with SOTA result, but to analyze the performance of federated learning. 
Therefore, we train a CNN with a simple architecture. The example script is e.g.

`python train_mnist_cnns.py --create_gnn --aggrPergraph 10 --modeslPeraggr 5 --local_epoch 5 --global_epoch 20 --model_perturb label --ascent_steps 3 --perturb_rate 0.5 --seed 42`

which will give a set of models trained on different digits with given initialization seeds. The trained models will be used as a dataset.

For FEMNIST dataset, we use the [LEAF](https://github.com/TalwalkarLab/leaf) repo, please see to the official repo and create FEMNIST dataset. 
After that, simply copy the `train` and `test` folder under `./datasets/FEMNIST`.

## Train GNN and MLP baseline
Once you have created the graph datasets, you can train a heterogeneous GNN on them.
Simply run `train_gnn.py` or `mlp_baseline.py` with deep learning hyperparameters. Don't forget to include your created datasets :)

## Dynamic filtering
After training the GNN, we can filter out malicious parameter nodes. For comparison with/without filtering, run e.g.

`python train_perturb_mnist_cnns.py --global_epoch 20 --seed 42 --model_perturb label --ascent_steps 3 --perturb_rate 0.5`

`python train_perturb_mnist_cnns.py --global_epoch 20 --seed 42 --model_perturb label --ascent_steps 3 --perturb_rate 0.5 --filter_models  --modelpath trained_gnns/gnn0.pt --aggrPergraph 10 --modeslPeraggr 5`
