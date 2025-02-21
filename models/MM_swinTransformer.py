from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
#print(sys.path)

import numpy as np
import cv2
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from config import _C as MC
from modelutils import load_pretrained
from models.build import build_model
from models.tab_transformer_noMLP import TabTransformer
from models.cross_attention import CrossTransformer, SwinCrossTransformer, PatchMerging, CrossTransformer_meta

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class MM_swinTransformer(nn.Module):
    def __init__(self, num_classes=5):
        super(MM_swinTransformer, self).__init__()
        self.cli_swin = build_model(MC)
        load_pretrained(MC, self.cli_swin)

        self.der_swin = build_model(MC)
        load_pretrained(MC, self.der_swin)

        self.meta_subnet = TabTransformer(
            categories = (3,3,9,2,3),      # tuple containing the number of unique values within each category
            num_continuous = 0,                # number of continuous values
            dim = 32,                           # dimension, paper set at 32
            #dim_out = 1,                        # binary prediction, but could be anything
            depth = 6,                          # depth, paper recommended 6
            heads = 8,                          # heads, paper recommends 8
            attn_dropout = 0.1,                 # post-attention dropout
            ff_dropout = 0.1,                   # feed forward dropout
            #mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
            #mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            #continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.num_classes = num_classes
        hidden_num = 128 * 3
        self.fusion_head = nn.Sequential(
            nn.Linear(1696 * 1, hidden_num),
            nn.ReLU(inplace=True)
        )

        self.fc_diag = nn.Linear(hidden_num, self.num_classes)
        self.fc_pn = nn.Linear(hidden_num, 3)
        self.fc_bwn = nn.Linear(hidden_num, 2)
        self.fc_vs = nn.Linear(hidden_num, 3)
        self.fc_pig = nn.Linear(hidden_num, 3)
        self.fc_str = nn.Linear(hidden_num, 3)
        self.fc_dag = nn.Linear(hidden_num, 3)
        self.fc_rs = nn.Linear(hidden_num, 2)

    def forward(self, meta_cat, meta_con, cli_x, der_x):
        meta_h = self.meta_subnet(meta_cat,meta_con)
        cli_h = self.cli_swin.patch_embed(cli_x)
        cli_h = self.cli_swin.pos_drop(cli_h)
        for i in range(4):  # 4 transformer layers in the cli_swin model
            cli_h = self.cli_swin.layers[i](cli_h)
            #print("cli_h %s for layer %s" % (cli_h.size(), i))

        cli_h = self.cli_swin.norm(cli_h)

        der_h = self.der_swin.patch_embed(der_x)
        der_h = self.der_swin.pos_drop(der_h)
        for i in range(4):  # 4 transformer layers in the der_swin model
            der_h = self.der_swin.layers[i](der_h)
            #print("Der_h %s for layer %s" % (der_h.size(), i))

        der_h = self.der_swin.norm(der_h)

        cli_f = self.avgpool(cli_h.transpose(1, 2)).view(cli_h.size(0), -1)
        der_f = self.avgpool(der_h.transpose(1, 2)).view(der_h.size(0), -1)
        #print('cli_f size', cli_f.size())
        #print('der_f size', der_f.size())
        #print('meta_f size', meta_h.size())
        feature_f = torch.cat([cli_f, der_f, meta_h], dim=-1)
        #feature_f = torch.cat([cli_f, der_f], dim=-1)
        #print('feature f size', feature_f.size())
        x = self.fusion_head(feature_f)
        #print('fusion head size', x.size())

        diag = self.fc_diag(x)
        #print('diag head size',diag.size())
        pn = self.fc_pn(x)
        #print('pn head size',pn.size())
        bwv = self.fc_bwn(x)
        #print('bwv head size',bwv.size())
        vs = self.fc_vs(x)
        #print('vs head size',vs.size())
        pig = self.fc_pig(x)
        #print('pig head size',pig.size())
        str = self.fc_str(x)
        #print('str head size',str.size())
        dag = self.fc_dag(x)
        #print('dag head size',dag.size())
        rs = self.fc_rs(x)
        #print('rs head size',rs.size())

        return [diag, pn, bwv, vs, pig, str, dag, rs]
    
    def criterion(self, logit, truth,weight=None):
        if weight == None:
            loss = F.cross_entropy(logit, truth)
        else:
            loss = F.cross_entropy(logit, truth,weight=weight)

        return loss
    
def main():
    in_size = [20, 224, 224]
    hidden_size = [16, 64, 64]
    out_size = [1, 8, 8]
    dropouts = [0.5, 0.5, 0.5]

    model = MM_swinTransformer(num_classes=5)
    print(model)

    cli = torch.randn(4, 3, 224, 224)
    der = torch.randn(4, 3, 224, 224)
    meta_cat = torch.cat([
        torch.randint(0, 3, (4, 1)),  # level_of_diagnostic_difficulty
        torch.randint(0, 3, (4, 1)),  # elevation
        torch.randint(0, 9, (4, 1)),  # location
        torch.randint(0, 2, (4, 1)),  # sex
        torch.randint(0, 3, (4, 1))   # management
    ], dim=1)    # category values, from 0 - max number of categories, in the order as passed into the constructor above
    meta_con = torch.randn(0, 0)               # For the derm7pt usecase, number of continous values is zero (in general assume continuous values are already normalized individually)
    # Forward pass
    model_output = model(meta_cat, meta_con, cli, der)
    print(model_output)

if __name__ == '__main__':
    main()