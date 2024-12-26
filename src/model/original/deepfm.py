import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any

from src.model.original.fm import FM
from src.model.original.layers import FeatureEmbedding, FM_Linear, MLP


class DeepFM(pl.LightningModule):
    def __init__(self, args, field_dims):
        super(DeepFM, self).__init__()
        self.linear = FM_Linear(args, field_dims)
        self.bceloss = nn.BCEWithLogitsLoss()
        self.lr = args.lr
        self.args = args
        self.sig = nn.Sigmoid()
        self.lastlinear = nn.Linear(3,1)
        
        self.fm = FM(args, field_dims)
        self.embedding = FeatureEmbedding(args, field_dims)
        self.embed_output_dim = len(field_dims) * args.emb_dim + args.cont_dims * args.emb_dim
        self.mlp = MLP(args, self.embed_output_dim)

    def l2norm(self):
        reg = 0
        for param in self.linear.parameters():
            reg += torch.norm(param)**2
        for param in self.embedding.parameters():
            reg += torch.norm(param)**2
        for param in self.mlp.parameters():
            reg += torch.norm(param)**2
        return reg*self.args.weight_decay

    def loss(self, y_pred, y_true, c_values):
        bce = self.bceloss(y_pred, y_true.float())
        weighted_bce = c_values * bce
        loss_y = weighted_bce.mean() + self.l2norm()
        return loss_y
    
    def forward(self, x, x_cont):
        # FM part, here, x_hat means another arbritary input of data, for combining the results
        embed_x = self.embedding(x)
        fm_part, cont_emb, lin_term, inter_term = self.fm.forward(x=x, x_cont=x_cont, emb_x=embed_x)
        if self.args.cont_dims!=0: # 기존에는 if cont_emb is None이었음
            embed_x = torch.cat((embed_x, cont_emb), 1)
        feature_number = embed_x.shape[1]
        embed_x = embed_x.view(-1, feature_number * self.args.emb_dim)
        
        deep_part = self.mlp(embed_x)

        lin_term = self.sig(lin_term)
        inter_term = self.sig(inter_term)
        deep_part = self.sig(deep_part)

        outs = torch.cat((lin_term, inter_term, deep_part), 1)
        y_pred = self.lastlinear(outs).squeeze(1)

        return y_pred

    def training_step(self, batch, batch_idx):
        x, x_cont, y, c_values = batch
        y_pred = self.forward(x, x_cont)
        loss_y = self.loss(y_pred, y,c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer