from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from IPython.display import Image
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from config import _C as MC
from modelutils import load_pretrained
from models.build import build_model
from models.tab_transformer_noMLP import TabTransformer
from models.cross_attention import CrossTransformer, SwinCrossTransformer, PatchMerging, CrossTransformer_meta

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MM_Transformer(nn.Module):
    def __init__(self, num_classes=5, cross_attention_depths=[1, 2, 3, 4, 2], hidden_dim=256, meta_dim=32):
        super(MM_Transformer, self).__init__()
        assert len(cross_attention_depths) == 5, "cross_attention_depths must be a list of five integers"

        self.cli_swin = build_model(MC).to(device)
        load_pretrained(MC, self.cli_swin)

        self.der_swin = build_model(MC).to(device)
        load_pretrained(MC, self.der_swin)

        self.meta_subnet = TabTransformer(
            categories=(3, 3, 9, 2, 3),
            num_continuous=0,
            dim=meta_dim,
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
        ).to(device)

        # cross attention
        mlp_ratio = 4.0
        attn_drop = 0.1
        proj_drop = 0.1
        drop_path = 0.1
       
        self.cross_attention_1 = SwinCrossTransformer(
            x_dim=192, c_dim=192, depth=cross_attention_depths[0], input_resolution=[56 // 2, 56 // 2],
            num_heads=6, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path
        )
        
        self.cross_attention_2 = SwinCrossTransformer(
            x_dim=384, c_dim=384, depth=cross_attention_depths[1], input_resolution=[56 // 4, 56 // 4],
            num_heads=6, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path
        )
        
        self.cross_attention_3 = SwinCrossTransformer(
            x_dim=768, c_dim=768, depth=cross_attention_depths[2], input_resolution=[56 // 8, 56 // 8],
            num_heads=6, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path
        )
        self.cross_attention_4 = SwinCrossTransformer(
            x_dim=768, c_dim=768, depth=cross_attention_depths[3], input_resolution=[56 // 8, 56 // 8],
            num_heads=12, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1).to(device)
        self.num_classes = num_classes

        self.fusion_head_cli = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(inplace=True)
        ).to(device)
        self.fusion_head_der = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(inplace=True)
        ).to(device)
        self.fusion_head_meta = nn.Sequential(
            nn.Linear(160, hidden_dim),
            nn.ReLU(inplace=True)
        ).to(device)

        self.fusion_meta = CrossTransformer_meta(x_dim=hidden_dim, c_dim=2 * hidden_dim, depth=cross_attention_depths[4], num_heads=8).to(device)

        self.fc_diag = nn.Linear(hidden_dim, self.num_classes).to(device)
        self.fc_pn = nn.Linear(hidden_dim, 3).to(device)
        self.fc_bwn = nn.Linear(hidden_dim, 2).to(device)
        self.fc_vs = nn.Linear(hidden_dim, 3).to(device)
        self.fc_pig = nn.Linear(hidden_dim, 3).to(device)
        self.fc_str = nn.Linear(hidden_dim, 3).to(device)
        self.fc_dag = nn.Linear(hidden_dim, 3).to(device)
        self.fc_rs = nn.Linear(hidden_dim, 2).to(device)

    def forward(self, meta_cat, meta_con, cli_x, der_x):
        # Move all input tensors to the GPU
        meta_cat = meta_cat.to(device)
        meta_con = meta_con.to(device)
        cli_x = cli_x.to(device)
        der_x = der_x.to(device)

        meta_h = self.meta_subnet(meta_cat, meta_con)

        cli_h = self.cli_swin.patch_embed(cli_x)
        cli_h = self.cli_swin.pos_drop(cli_h)
        der_h = self.der_swin.patch_embed(der_x)
        der_h = self.der_swin.pos_drop(der_h)

        cli_0 = self.cli_swin.layers[0](cli_h)
        der_0 = self.der_swin.layers[0](der_h)
        cli_0, der_0 = self.cross_attention_1(cli_0, der_0)

        cli_1 = self.cli_swin.layers[1](cli_0)
        der_1 = self.der_swin.layers[1](der_0)
        cli_1, der_1 = self.cross_attention_2(cli_1, der_1)

        cli_2 = self.cli_swin.layers[2](cli_1)
        der_2 = self.der_swin.layers[2](der_1)
        cli_2, der_2 = self.cross_attention_3(cli_2, der_2)

        cli_3 = self.cli_swin.layers[3](cli_2)
        der_3 = self.der_swin.layers[3](der_2)
        cli_3, der_3 = self.cross_attention_4(cli_3, der_3)

        cli_h = self.cli_swin.norm(cli_3)
        der_h = self.der_swin.norm(der_3)
        cli_f = self.avgpool(cli_h.transpose(1, 2))
        der_f = self.avgpool(der_h.transpose(1, 2))
        cli_f = torch.flatten(cli_f, 1)
        der_f = torch.flatten(der_f, 1)
        der_f = self.fusion_head_der(der_f)
        cli_f = self.fusion_head_cli(cli_f)

        feature_f = torch.cat([cli_f, der_f], dim=-1)
        meta_f = self.fusion_head_meta(meta_h)
        x = self.fusion_meta(meta_f, feature_f)
        
        diag = self.fc_diag(x)
        pn = self.fc_pn(x)
        bwv = self.fc_bwn(x)
        vs = self.fc_vs(x)
        pig = self.fc_pig(x)
        str = self.fc_str(x)
        dag = self.fc_dag(x)
        rs = self.fc_rs(x)

        return [diag, pn, bwv, vs, pig, str, dag, rs]
    
    def list_layers(self):
        for name, module in self.cli_swin.named_children():
            print(f"Layer name: {name}, Layer type: {type(module)}")

    def model_summary(self):
        dummy_meta_cat = torch.cat([
            torch.randint(0, 3, (4, 1)),
            torch.randint(0, 3, (4, 1)),
            torch.randint(0, 9, (4, 1)),
            torch.randint(0, 2, (4, 1)),
            torch.randint(0, 3, (4, 1))
        ], dim=1).to(device)
        dummy_meta_con = torch.empty((4, 0)).to(device)
        dummy_cli = torch.randn(4, 3, 224, 224).to(device)
        dummy_der = torch.randn(4, 3, 224, 224).to(device)
        summary(self, input_data=(dummy_meta_cat, dummy_meta_con, dummy_cli, dummy_der), col_names=["input_size", "output_size", "num_params", "trainable"])

def main():
    model = MM_Transformer(num_classes=5, cross_attention_depths=[1, 1, 1, 1, 1]).to(device)
    
    cli = torch.randn(4, 3, 224, 224).to(device)
    der = torch.randn(4, 3, 224, 224).to(device)
    meta_cat = torch.cat([
        torch.randint(0, 3, (4, 1)),
        torch.randint(0, 3, (4, 1)),
        torch.randint(0, 9, (4, 1)),
        torch.randint(0, 2, (4, 1)),
        torch.randint(0, 3, (4, 1))
    ], dim=1).to(device)
    meta_con = torch.empty((4, 0)).to(device)
    print(device)
    model_output = model(meta_cat, meta_con, cli, der)
    print(model_output)

    model.model_summary()

if __name__ == "__main__":
    main()