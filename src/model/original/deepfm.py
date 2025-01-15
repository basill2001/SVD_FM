import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any

import torch
from src.model.original.fm import FM
from src.model.original.layers import FeatureEmbedding, FeatureEmbedding, FM_Linear, MLP


class DeepFM(pl.LightningModule):
    def __init__(self, args, field_dims):
        super(DeepFM, self).__init__()
        self.args = args
        self.lr = args.lr
        self.field_dims = field_dims
        self.linear = FM_Linear(args, field_dims)
        self.fm = FM(args, field_dims)
        self.embedding = FeatureEmbedding(args, field_dims)
        self.embed_output_dim = len(field_dims) * args.emb_dim + args.cont_dims * args.emb_dim
        self.mlp = MLP(args, self.embed_output_dim)
        self.bceloss=nn.BCEWithLogitsLoss() # since bcewith logits is used, we don't need to add sigmoid layer in the end

        self.sig = nn.Sigmoid()
        self.lastlinear = nn.Linear(3,1)

    def l2norm(self):
        reg = 0
        for param in self.linear.parameters():
            reg += torch.norm(param)**2
        for param in self.embedding.parameters():
            reg += torch.norm(param)**2
        for param in self.mlp.parameters():
            reg += torch.norm(param)**2
        return reg*self.args.weight_decay


    def mse(self, y_pred, y_true):
        return self.bceloss(y_pred, y_true.float())
    
    def deep_part(self, x):
        return self.mlp(x)

    def loss(self, y_pred, y_true, c_values):
        mse =self.bceloss(y_pred, y_true.float())
        weighted_bce = c_values * mse
        loss_y = weighted_bce.mean()
        loss_y += self.l2norm()

        return loss_y
    
    def forward(self, x, x_cont):
        # FM part, here, x_hat means another arbritary input of data, for combining the results. 
        
        embed_x = self.embedding(x)
        fm_part, cont_emb, lin_term, inter_term = self.fm.forward(x, x_cont, embed_x)
        
        if cont_emb is not None:
            embed_x = torch.cat((embed_x, cont_emb), 1)
        feature_number = embed_x.shape[1]
        embed_x = embed_x.view(-1, feature_number * self.args.emb_dim)

        new_x = embed_x
        deep_part = self.mlp(new_x)

        lin_term = self.sig(lin_term)
        inter_term = self.sig(inter_term)
        deep_part = self.sig(deep_part)
        outs = torch.cat((lin_term,inter_term), 1)
        outs = torch.cat((outs,deep_part), 1)
        y_pred = self.lastlinear(outs).squeeze(1)

        return y_pred

    def training_step(self, batch, batch_idx):
        x, x_cont, y, c_values = batch
        y_pred = self.forward(x,x_cont)
        loss_y = self.loss(y_pred, y,c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer