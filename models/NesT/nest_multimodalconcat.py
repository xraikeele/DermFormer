import timm
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from  models.tab_transformer_noMLP import TabTransformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the multimodal NesT model
class nest_MMC(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=128, model_name='jx_nest_tiny', num_heads=2, dropout_rate=0.5, meta_dim=32):
        super(nest_MMC, self).__init__()

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

        self.meta_reduce = nn.Sequential(
            nn.Linear(160, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )   
        # Linear layer to reduce concatenated feature size
        self.feature_reduce = nn.Linear(hidden_dim * 7, hidden_dim)  # 896 -> 128 (or other desired size)

        # Define classification heads for different outputs
        self.num_classes = num_classes
        self.fc_diag = nn.Linear(hidden_dim, self.num_classes)
        self.fc_pn = nn.Linear(hidden_dim, 3)
        self.fc_bwn = nn.Linear(hidden_dim, 2)
        self.fc_vs = nn.Linear(hidden_dim, 3)
        self.fc_pig = nn.Linear(hidden_dim, 3)
        self.fc_str = nn.Linear(hidden_dim, 3)
        self.fc_dag = nn.Linear(hidden_dim, 3)
        self.fc_rs = nn.Linear(hidden_dim, 2)

    def forward(self, meta_cat, meta_con, cli_x, der_x):
        # Forward pass through both branches
        cli_features = self.cli_nest(cli_x)  # Features from the first branch (cli)
        der_features = self.der_nest(der_x)  # Features from the second branch (der)

        # Meta features through TabTransformer
        meta_cat = meta_cat.long()
        meta_con = meta_con.long()
        meta_features = self.meta_subnet(meta_cat, meta_con)
        #print(f"meta_features shape: {meta_features.shape}")
        meta_features = self.meta_reduce(meta_features)
        #print(f"meta_features shape: {meta_features.shape}")
        #x = meta_features
        # Concatenate outputs from both branches
        x = torch.cat((der_features, cli_features, meta_features), dim=1)
        #x = torch.cat((der_features, meta_features), dim=1)
        #print(f"x_features shape: {x.shape}")
        x = self.feature_reduce(x)
        #print(f"x_features shape: {x.shape}")
        # Pass through classification heads
        diag = self.fc_diag(x)
        pn = self.fc_pn(x)
        bwn = self.fc_bwn(x)
        vs = self.fc_vs(x)
        pig = self.fc_pig(x)
        str_ = self.fc_str(x)
        dag = self.fc_dag(x)
        rs = self.fc_rs(x)

        # Return a list of classification outputs
        return [diag, pn, bwn, vs, pig, str_, dag, rs]
    
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
    model = nest_MMC(num_classes=5, hidden_dim=128).to(device)
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
