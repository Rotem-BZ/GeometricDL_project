import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MessagePassing, SAGEConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        # losses_sum = torch.tensor(0.0, device=self.device)  # used for backward
        x = batch.x
        y = batch.y.view(-1)[:batch.batch_size]
        edge_index = batch.edge_index
        for layer, classifier, bn_layer in zip(self.layers[:-1], self.classifiers[:-1], self.bn_layers):
            # zip doesn't take last element of layers & classifiers
            x = layer(x, edge_index)
            x = bn_layer(x)
            pred = classifier(x[:batch.batch_size])
            loss = self.loss_func(pred, y)
            # losses_sum += loss
            losses_list.append(loss)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        pred = self.classifiers[-1](x[:batch.batch_size])
        loss = self.loss_func(pred, y)
        # losses_sum += loss
        losses_list.append(loss)
        losses = torch.stack(losses_list)
        losses_sum = torch.sum(losses)
        minimal_loss = torch.min(losses)

        self.log(f'{state}_loss', losses_sum.item(), batch_size=batch.batch_size)
        # print(f"loss is {losses_sum.item()} shape {losses.shape}")
        self.log(f'minimal_{state}_loss', minimal_loss.item(), batch_size=batch.batch_size)
        return {'loss': losses_sum, 'minimal_loss': minimal_loss}
        # return ('loss': loss)

    def training_step(self, batch, batch_idx):
        return self(batch, batch_idx, 'train')

    def epoch_end(self, outputs, stage):
        if outputs:
            for metric in outputs[0].keys():
                if self.trainer.global_step == 0:
                    wandb.define_metric(f'{stage}_epoch_{metric}', summary='min')
                losses = torch.stack([output[metric] for output in outputs])
                mean_loss = torch.mean(losses)
                self.log(f'{stage}_epoch_{metric}', mean_loss.item())

    def training_epoch_end(self, outputs) -> None:
        self.epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs) -> None:
        self.epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs) -> None:
        # losses = torch.stack([output['loss'] for output in outputs])
        # self.logger.experiment.summary['test_loss'] = torch.mean(losses)
        self.epoch_end(outputs, 'test')


    def validation_step(self, batch, batch_idx):
        return self(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.03)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2, min_lr=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

