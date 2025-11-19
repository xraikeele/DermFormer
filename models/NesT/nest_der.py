from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import timm
import torch 
import torch.nn as nn
from torchinfo import summary
from models.tabtransformer.tab_transformer_noMLP import TabTransformer
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable parameter
    
    def forward(self, x1, x2):
        # Generate attention weights
        attn_weights = torch.sigmoid(self.fc1(x1) + self.fc2(x2))
        # Learnable alpha determines the weighting between the two inputs
        return self.alpha * (attn_weights * x1 + (1 - attn_weights) * x2)
    
class nest_der(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=128, model_name='jx_nest_small', num_heads=2, dropout_rate=0.5, meta_dim=32):
        super(nest_der, self).__init__()

        # Backbone models for clinical and dermoscopy branches
        self.cli_nest = timm.create_model(model_name, pretrained=True)
        self.der_nest = timm.create_model(model_name, pretrained=True)

        # Modify the classifier layers to extract intermediate features
        num_features = self.cli_nest.get_classifier().in_features
        self.cli_nest.head = nn.Identity()
        self.der_nest.head = nn.Identity()

        self.meta_subnet = TabTransformer(
            categories=(3, 3, 9, 2, 3),
            num_continuous=0,
            dim=meta_dim,
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )
        
        # Adjust the feature reduction layers
        self.cli_reduce = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.der_reduce = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.meta_reduce = nn.Sequential(
            nn.Linear(160, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )   
        # Attention layer
        self.attention = AttentionLayer(hidden_dim)

        # Attention layer 2
        self.attention_2 = AttentionLayer(hidden_dim)

        # Feature reduction layer for concatenated features
        self.feature_reduce = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Classification heads
        self.fc_diag = nn.Linear(hidden_dim, num_classes)
        self.fc_pn = nn.Linear(hidden_dim, 3)
        self.fc_bwn = nn.Linear(hidden_dim, 2)
        self.fc_vs = nn.Linear(hidden_dim, 3)
        self.fc_pig = nn.Linear(hidden_dim, 3)
        self.fc_str = nn.Linear(hidden_dim, 3)
        self.fc_dag = nn.Linear(hidden_dim, 3)
        self.fc_rs = nn.Linear(hidden_dim, 2)

    def forward(self, meta_cat, meta_con, cli_x, der_x):
        # Process through both branches
        der_features = self.der_nest(der_x)
        
        x = self.feature_reduce(der_features)

        # Pass through classification heads
        diag = self.fc_diag(x)
        pn = self.fc_pn(x)
        bwv = self.fc_bwn(x)
        vs = self.fc_vs(x)
        pig = self.fc_pig(x)
        str = self.fc_str(x)
        dag = self.fc_dag(x)
        rs = self.fc_rs(x)

        return [diag, pn, bwv, vs, pig, str, dag, rs]
    
    def criterion(self, logit, truth,weight=None):
        if weight == None:
            loss = F.cross_entropy(logit, truth)
        else:
            loss = F.cross_entropy(logit, truth,weight=weight)

        return loss

def main():
    model_names_with_pretrained = timm.list_models('*nest*', pretrained=True)
    print(model_names_with_pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nest_der(num_classes=5, hidden_dim=128).to(device)
    print(model)
    cli_x = torch.randn(16, 3, 224, 224).to(device)
    der_x = torch.randn(16, 3, 224, 224).to(device)
    meta_cat = torch.cat([
        torch.randint(0, 3, (16, 1)),
        torch.randint(0, 3, (16, 1)),
        torch.randint(0, 9, (16, 1)),
        torch.randint(0, 2, (16, 1)),
        torch.randint(0, 3, (16, 1))
    ], dim=1).to(device)
    meta_con = torch.empty((4, 0)).to(device)
    outputs = model(meta_cat, meta_con, cli_x, der_x)
    for i, output in enumerate(outputs):
        print(f"Output {i}: {output.shape}")

if __name__ == "__main__":
    main()