from pathlib import Path

import torch
import pickle
# import joblib
import os
from os.path import join, isdir
from typing import Optional
import time
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import wandb

from network import OurGNN

DATA_FOLDER = 'saved_data'

DEPTH_MODEL_PATH = 'depth_model.pkl'


class OurDataModule(pl.LightningDataModule):
    def __init__(self, max_depth, batch_size, neighbor_limit, path, centrality_path, device):
        # TODO: implement for cuda
        super(OurDataModule, self).__init__()
        device = device['accelerator']
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.neighbor_limit = neighbor_limit
        if device == 'cpu':
            self.num_workers = os.cpu_count() - 2
        elif device == 'gpu':
            self.num_workers = 5
        else:
            raise ValueError
        self.path = path
        self.data, self.num_classes, self.train_idx, self.valid_idx, self.test_idx, self.export_idx = [None] * 6
        self.centrality_df = pd.read_csv(centrality_path, index_col=0)

    def prepare_data(self) -> None:
        # download
        _ = PygNodePropPredDataset(name='ogbn-arxiv', root=self.path)

    def setup(self, stage: Optional[str] = None):
        # stage variable is not used
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=self.path)
        self.data = dataset[0]
        self.data.n_ids = torch.arange(self.data.num_nodes)
        self.data.extracted_features = torch.tensor(self.centrality_df.values)
        self.num_classes = dataset.num_classes
        split_idx = dataset.get_idx_split()

        self.train_idx = split_idx['train']
        self.valid_idx = split_idx['valid']
        self.test_idx = split_idx['test'][:10000]
        self.export_idx = split_idx['test'][10000:]

    def train_dataloader(self):
        return NeighborLoader(self.data, input_nodes=self.train_idx,
                              shuffle=True, num_workers=self.num_workers,
                              batch_size=self.batch_size, num_neighbors=[self.neighbor_limit] * self.max_depth)

    def val_dataloader(self):
        return NeighborLoader(self.data, input_nodes=self.valid_idx,
                              shuffle=False, num_workers=self.num_workers,
                              batch_size=self.batch_size, num_neighbors=[self.neighbor_limit] * self.max_depth)

    def test_dataloader(self):
        return NeighborLoader(self.data, input_nodes=self.test_idx,
                              shuffle=False, num_workers=self.num_workers,
                              batch_size=self.batch_size, num_neighbors=[self.neighbor_limit] * self.max_depth)

    def depth_dataloader(self):
        return NeighborLoader(self.data, input_nodes=self.export_idx,
                              shuffle=False, num_workers=self.num_workers,
                              batch_size=self.batch_size, num_neighbors=[self.neighbor_limit] * self.max_depth)


def create_data(network: OurGNN, batch_size, neighbor_limit, data_path, cent_path, device):
    X = []
    Y = []

    data_module = OurDataModule(network.max_depth, batch_size, neighbor_limit, data_path, cent_path, device)
    data_module.setup()
    loader = data_module.depth_dataloader()

    with torch.inference_mode():
        for batch in tqdm(loader):
            features = batch.extracted_features[:batch.batch_size]  # should be (num_nodes, num_features)
            targets = network.optimal_depth(batch)
            X.append(features)
            Y += targets.tolist()

    X = torch.vstack(X)
    Y = torch.tensor(Y)
    features_path = join(DATA_FOLDER, f'{network.max_depth}_layers_features.pt')
    labels_path = join(DATA_FOLDER, f'{network.max_depth}_layers_optimal_depths.pt')
    if not isdir(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)
    with open(features_path, 'wb') as file:
        torch.save(X, file)
    with open(labels_path, 'wb') as file:
        torch.save(Y, file)


def main():
    max_depth = None
    batch_size = 1024
    epochs = 150
    neighbor_limit = -1
    embed_dim = 256

    debug_mode = True
    device = {'accelerator': 'cpu'}
    # device = {'accelerator': 'gpu', 'devices': [1]}
    data_path = '/data/user-data/niv.ko/networks'

    if debug_mode:
        wandb_logger = None
        epochs = 1
    else:
        wandb_logger = WandbLogger(project="geometric_DL", log_model="all", name=f'depth_{max_depth}_epochs_{epochs}')

    cent_path = '/home/niv.ko/gnn_depth/measures/cent_data.csv'
    # data_module = OurDataModule(max_depth, batch_size, neighbor_limit, data_path, device=device, centrality_path=cent_path)
    # data_module.setup()

    ###########################################################################
    # train model
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor="val_epoch_minimal_loss", mode="min"))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    callbacks.append(EarlyStopping(monitor="val_epoch_minimal_loss", mode="min", patience=20))
    # model = OurGNN(max_depth, data_module.data.x.shape[1], embed_dim, data_module.num_classes)
    # trainer = pl.Trainer(logger=wandb_logger, max_epochs=epochs, **device, callbacks=callbacks,
    #                      check_val_every_n_epoch=2)
    # trainer.fit(model, data_module)
    # trainer.test(model, data_module)
    ###########################################################################

    # ###########################################################################
    # create optimal-depth data
    # model = OurGNN(max_depth, data_module.data.x.shape[1], embed_dim, data_module.num_classes)
    # model.load_from_checkpoint('checkpoints/epoch=0-step=89-v1.ckpt')
    # model = OurGNN.load_from_checkpoint('checkpoints/epoch=0-step=89-v1.ckpt')
    # reference can be retrieved in artifacts panel
    # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
    checkpoint_reference = 'niv-ko/geometric_DL/model-d33mm2v9:v16'

    # download checkpoint locally (if not already cached)
    run = wandb.init(project="geometric_DL")
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()

    # load checkpoint
    model = OurGNN.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
    print('hi')
    create_data(model, batch_size, neighbor_limit, data_path, cent_path, device)
    # ###########################################################################
    #
    # ###########################################################################
    # # train decision tree for optimal depth
    # train_depth_regressor()     # note: doesn't require the original data
    # ###########################################################################


if __name__ == '__main__':
    main()
