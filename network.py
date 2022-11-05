import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MessagePassing, SAGEConv

import pytorch_lightning as pl


class OurGNN(pl.LightningModule):
    def __init__(self, max_depth: int, in_dim: int, embed_dim: int, out_classes: int):
        super(OurGNN, self).__init__()
        self.max_depth = max_depth
        dims = [in_dim] + [embed_dim] * max_depth
        layers = []
        for layer_in_dim, layer_out_dim in zip(dims[:-1], dims[1:]):
            layers.append(SAGEConv(layer_in_dim, layer_out_dim))
        self.layers = nn.ModuleList(layers)
        self.classifiers = nn.ModuleList(nn.Linear(embed_dim, out_classes) for _ in range(max_depth))
        self.bn_layers = nn.ModuleList(nn.BatchNorm1d(embed_dim) for _ in range(max_depth - 1))
        self.dropout = nn.Dropout()
        self.loss_func = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def optimal_depth(self, batch):
        """
        This will only be used in inference after the classifiers are trained. Extracts the optimal depth for
        each given node.
        :param batch.x: node feature matrix: (num_nodes, num_features)
        :param batch.edge_index: SOO format edge existence matrix
        :return:
        the index of the optimal depth for each node (classifier with lowest loss)
        """
        # print("entered forward")
        x = batch.x
        losses = []
        for layer, classifier, bn_layer in zip(self.layers[:-1], self.classifiers[:-1], self.bn_layers):
            # zip doesn't take last element of layers & classifiers
            x = layer(x, batch.edge_index)
            x = bn_layer(x)
            pred = classifier(x[:batch.batch_size])
            unreduced_loss = F.cross_entropy(pred, batch.y.view(-1)[:batch.batch_size], reduction='none')
            losses.append(unreduced_loss)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x, batch.edge_index)
        pred = self.classifiers[-1](x[:batch.batch_size])
        unreduced_loss = F.cross_entropy(pred, batch.y.view(-1)[:batch.batch_size], reduction='none')
        losses.append(unreduced_loss)

        loss_mat = torch.vstack(losses)     # (max_depth, num_nodes)
        optimal_depths = torch.argmin(loss_mat, dim=0)      # (num_nodes)
        return optimal_depths

    def forward(self, batch, batch_idx, state: str):
        losses_list = []    # used for checkpointing
        losses_sum = torch.tensor(0.0)  # used for backward
        x = batch.x
        y = batch.y.view(-1)[:batch.batch_size]
        edge_index = batch.edge_index
        for layer, classifier, bn_layer in zip(self.layers[:-1], self.classifiers[:-1], self.bn_layers):
            # zip doesn't take last element of layers & classifiers
            x = layer(x, edge_index)
            x = bn_layer(x)
            pred = classifier(x[:batch.batch_size])
            loss = self.loss_func(pred, y)
            losses_sum += loss
            losses_list.append(loss.item())
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        pred = self.classifiers[-1](x[:batch.batch_size])
        loss = self.loss_func(pred, y)
        losses_sum += loss
        losses_list.append(loss.item())
        self.log(f'{state}_loss', losses_sum.item())
        print(f"loss is {losses_sum.item()}")
        self.log(f'minimal_{state}_loss', min(losses_list))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch, batch_idx, 'val')

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer

