from typing import Any
import torch
import torch.nn as nn
from src.model.layers import FeatureEmbedding, FM_Linear, FM_Interaction

import pytorch_lightning as pl

class FM(pl.LightningModule):
    def __init__(self, args, field_dims):
        super(FM, self).__init__()
        self.args = args
        self.embedding = FeatureEmbedding(args, field_dims)
        self.linear = FM_Linear(args, field_dims)
        self.interaction = FM_Interaction(args)
        self.bceloss = nn.BCEWithLogitsLoss()
        self.sig = nn.Sigmoid()
        self.last_linear = nn.Linear(2, 1)

    def l2norm(self):
        reg = 0
        for param in self.linear.parameters():
            reg += torch.norm(param)**2
        for param in self.embedding.parameters():
            reg += torch.norm(param)**2
        for param in self.interaction.parameters():
            reg += torch.norm(param)**2
        return reg * self.args.weight_decay

    def loss(self, y_pred, y_true, c_values):
        bce = self.bceloss(y_pred, y_true.float())
        weighted_bce = c_values * bce
        loss_y = weighted_bce.mean() + self.l2norm()
        return loss_y
    
    # def forward(self, x, x_cont, emb_x):
    #     "original forward"
    #     lin_term = self.linear(x=x, x_cont=x_cont, emb_x=None)
    #     inter_term = self.interaction(emb_x, x_cont, None, None)
    #     lin_term = self.sig(lin_term)
    #     inter_term = self.sig(inter_term)
    #     outs = torch.cat((lin_term, inter_term), 1)
    #     x = self.last_linear(outs)
    #     x = x.squeeze(1)
    #     return x
    
    def forward(self, x, x_cont, emb_x, svd_emb):
        "embeded forward"
        lin_term = self.linear(x=x, x_cont=x_cont, emb_x=svd_emb)
        inter_term = self.interaction(x_cont=x_cont, emb_x=emb_x, svd_emb=svd_emb)
        lin_term = self.sig(lin_term)
        inter_term = self.sig(inter_term)
        outs = torch.cat((lin_term, inter_term), 1)
        x = self.last_linear(outs)
        x = x.squeeze(1)
        return x

    def training_step(self, batch, batch_idx):
        if self.args.embedding_type=='original':
            x, x_cont, y, c_values = batch
            embed_x = self.embedding(x)
            y_pred = self.forward(x, x_cont, embed_x)
        else:
            x, svd_emb, ui, x_cont, y, c_values = batch
            embed_x = self.embedding(x)
            y_pred = self.forward(x, x_cont, embed_x, svd_emb)
        loss_y = self.loss(y_pred, y, c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) ->Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer