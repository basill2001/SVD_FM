import torch
import numpy as np
import json
import time
from copy import deepcopy

from torch.utils.data import Dataset
from src.util.negativesampler import NegativeSampler
import argparse
from src.data_util.customdataloader import CustomDataLoader
from src.data_util.SVDdataloader import SVDDataloader
from torch.utils.data import DataLoader
from src.data_util.datawrapper import DataWrapper
from src.model.original.fm import FactorizationMachine
from src.model.SVD_emb.svdfm import FactorizationMachineSVD
from src.model.SVD_emb.svddeepfm import DeepFMSVD   
from src.customtest import Emb_Test
from sklearn.preprocessing import LabelEncoder
from src.model.original.deepfm import DeepFM
from src.model.SVD import SVD
#copy
from src.util.preprocessor import Preprocessor
import pytorch_lightning as pl

# 인자 전달
parser = argparse.ArgumentParser()

parser.add_argument('--train_ratio', type=float, default=0.7, help='training ratio for any dataset')


# parser.add_argument('--num_factors', type=int, default=15, help='Number of factors for FM')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for fm training')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay(for both FM and autoencoder)')
# parser.add_argument('--num_epochs_ae', type=int, default=300,    help='Number of epochs')
parser.add_argument('--num_epochs_training', type=int, default=100,    help='Number of epochs')

parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
# parser.add_argument('--ae_batch_size', type=int, default=256, help='Batch size for autoencoder')

parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for dataloader')
parser.add_argument('--num_deep_layers', type=int, default=2, help='Number of deep layers')
parser.add_argument('--deep_layer_size', type=int, default=128, help='Size of deep layers')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save_model', type=bool, default=False)


parser.add_argument('--emb_dim', type=int, default=16, help='embedding dimension for DeepFM')
# parser.add_argument('--num_embedding', type=int, default=200, help='Number of embedding for autoencoder') 
parser.add_argument('--embedding_type', type=str, default='SVD', help='AE or SVD or original')
parser.add_argument('--model_type', type=str, default='fm', help='fm or deepfm')
parser.add_argument('--topk', type=int, default=5, help='top k items to recommend')
parser.add_argument('--fold', type=int, default=1, help='fold number for folded dataset')
parser.add_argument('--isuniform', type=bool, default=False, help='true if uniform false if not')
parser.add_argument('--ratio_negative', type=int, default=0.2, help='negative sampling ratio rate for each user')
# parser.add_argument('--auto_lr', type=float, default=0.01, help='autoencoder learning rate')
# parser.add_argument('--k', type=int, default=10, help='autoencoder k')
parser.add_argument('--num_eigenvector', type=int, default=16, help='Number of eigenvectors for SVD, note that this must be same as emb_dim')
parser.add_argument('--datatype', type=str, default="ml100k", help='ml100k or ml1m or shopping or goodbook or frappe')
parser.add_argument('--c_zeros', type=int, default=5, help='c_zero for negative sampling')
parser.add_argument('--cont_dims', type=int, default=0,help='continuous dimension(that changes for each dataset))')
parser.add_argument('--shopping_file_num', type=int, default=147,help='name of shopping file choose from 147 or  148 or 149')


args = parser.parse_args("")



def getdata(args):
    
    # get any dataset
    dataset = DataWrapper(args)

    train_df, test, item_info, user_info, ui_matrix = dataset.get_data()
    cat_cols, cont_cols = dataset.get_col_type()
    # those are basic dataframes that we can get from various datasets
    preprocessor = Preprocessor(args, train_df, test, user_info, item_info, ui_matrix, cat_cols, cont_cols)
    # preprocessor is a class that preprocesses dataframes and returns
    # : train_df, test_df, item_info, user_info, useritem_matrix, cat_columns, cont_columns, label_encoders, user_embedding, item_embedding
    return preprocessor


def trainer(args, data: Preprocessor):

    data.label_encode()
    items, conts = data.get_catcont_train()
    target, c = data.get_target_c()
    field_dims = data.get_field_dims()
    uidf = data.uidf.values

    # I know this is a bit inefficient to create all four classes for model, but I did this for simplicity
    if args.model_type=='fm' and args.embedding_type=='original':
        model = FactorizationMachine(args, field_dims)
        Dataset = CustomDataLoader(items, conts, target, c)

    elif args.model_type=='deepfm' and args.embedding_type=='original':
        model = DeepFM(args, field_dims)
        Dataset = CustomDataLoader(items, conts, target, c)

    elif args.model_type=='fm' and args.embedding_type=='SVD':
        model = FactorizationMachineSVD(args, field_dims)
        embs = conts[:, -args.num_eigenvector*2:]   # Here, numeighenvector*2 refers to embeddings for both user and item
        conts = conts[:, :-args.num_eigenvector*2]  # rest of the columns are continuous columns (e.g. age, , etc.)
        Dataset = SVDDataloader(items, embs, uidf, conts, target, c)

    elif args.model_type=='deepfm' and args.embedding_type=='SVD':
        model = DeepFMSVD(args, field_dims)
        embs = conts[:, -args.num_eigenvector*2:]   # Here, numeighenvector*2 refers to embeddings for both user and item
        conts = conts[:, :-args.num_eigenvector*2]  # rest of the columns are continuous columns (e.g. age, , etc.)
        Dataset = SVDDataloader(items, embs, uidf, conts, target, c)

    else:
        raise NotImplementedError
    
    
    # dataloaders
    dataloader = DataLoader(Dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)

    
    start = time.time()
    trainer = pl.Trainer(max_epochs=args.num_epochs_training)
    trainer.fit(model, dataloader)
    end = time.time()
    return model, end-start

if __name__=='__main__':
    args = parser.parse_args("")

    results={}

    data_info = getdata(args)

    print('model type is', args.model_type)
    print('embedding type is', args.embedding_type)
    model, timeee = trainer(args, data_info)
    test_time = time.time()
    tester = Emb_Test(args,model,data_info)


    if args.embedding_type=='SVD':
        result = tester.svdtest()
    else:
        result = tester.test()
    
    
    end_test_time = time.time()
    results[args.model_type+args.embedding_type]=result
    #results[md+embedding]=result
    print(results)
    print("time :", timeee)