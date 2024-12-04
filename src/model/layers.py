import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):

    def __init__(self, args, input_size):
        super(MLP, self).__init__()
        self.args = args
        self.deep_layers = nn.ModuleList()
        for i in range(args.num_deep_layers):
            self.deep_layers.append(nn.Linear(input_size, args.deep_layer_size))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.Dropout(p=0.2))
            input_size = args.deep_layer_size
        self.deep_output_layer = nn.Linear(input_size, 1)

    def forward(self, x):
        for layer in self.deep_layers:
            x = layer(x)
        x = self.deep_output_layer(x)
        return x

class FeatureEmbedding(nn.Module):

    def __init__(self, args, field_dims):
        super(FeatureEmbedding, self).__init__()
        if args.embedding_type=='original':
            self.embedding = nn.Embedding(sum(field_dims+1), args.emb_dim)
        else:
            self.embedding = nn.Embedding(sum(field_dims), args.emb_dim)
        self.field_dims = field_dims
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        x = self.embedding(x)
        return x

class FM_Linear(nn.Module):

    def __init__(self, args, field_dims):
        super(FM_Linear, self).__init__()
        if args.embedding_type=='original':
            self.linear = torch.nn.Embedding(sum(field_dims)+1, 1)
            self.w = nn.Parameter(torch.randn(args.cont_dims))
        else:
            self.linear = torch.nn.Embedding(sum(field_dims), 1)
            self.w = nn.Parameter(torch.randn(args.cont_dims-args.num_eigenvector*2))
        self.bias = nn.Parameter(torch.randn(1))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self.args = args
    
    def forward(self, x, x_cont, emb_x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        linear_term = self.linear(x)
        cont_linear = torch.matmul(x_cont, self.w).reshape(-1, 1)

        if self.args.embedding_type=='SVD' or self.args.embedding_type=='NMF':
            user_emb = emb_x[:, 0].unsqueeze(1).unsqueeze(1)
            item_emb = emb_x[:, self.args.num_eigenvector].unsqueeze(1).unsqueeze(1)
            nemb_x = torch.cat((user_emb, item_emb), 1)
            linear_term = torch.cat((linear_term, nemb_x), 1)
        x = torch.sum(linear_term, dim=1) + self.bias
        x = x + cont_linear
        return x

class FM_Interaction(nn.Module):

    def __init__(self, args):
        super(FM_Interaction, self).__init__()
        self.args = args
        if self.args.embedding_type=='original':
            self.v = nn.Parameter(torch.randn(args.cont_dims, args.emb_dim))
        else:
            self.v = nn.Parameter(torch.randn(args.cont_dims-args.num_eigenvector*2, args.emb_dim))
    
    def forward(self, x, x_cont, emb_x, svd_emb):
        """
        original일 경우 emb_x, x_cont만 사용
        SVD 또는 NMF일 경우 emb_x, svd_emb만 사용
        """
        x_cont = x_cont.unsqueeze(1)

        if self.args.embedding_type=='original':
            linear = torch.sum(x, 1)**2
            interaction = torch.sum(x**2, 1)
            if self.args.cont_dims!=0:
                cont_linear = torch.sum(torch.matmul(x_cont, self.v)**2, dim=1)
                linear = torch.cat((linear, cont_linear), 1)
                cont_interaction = torch.sum(torch.matmul(x_cont**2, self.v**2), 1, keepdim=True)
                interaction = torch.cat((interaction, cont_interaction.squeeze(1)), 1)
        else:
            user_emb = svd_emb[:, :self.args.num_eigenvector].unsqueeze(1)
            item_emb = svd_emb[:, self.args.num_eigenvector:].unsqueeze(1)
            cont = torch.matmul(x_cont, self.v)
            x = torch.cat((emb_x, user_emb, item_emb, cont), 1)
            linear = torch.sum(x, 1)**2
            interaction = torch.sum(x**2, 1)
            
        interaction = 0.5*torch.sum(linear-interaction, 1, keepdim=True)
        cont_emb = self.v.unsqueeze(0).repeat(x.shape[0], 1, 1)
        return interaction