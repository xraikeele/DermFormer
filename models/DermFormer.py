from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import timm
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from tab_transformer_noMLP import TabTransformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MMNestLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(MMNestLoss, self).__init__()

        # Initialize weights for each branch
        self.weights = nn.ParameterDict({
            'cli': nn.Parameter(torch.tensor(1.0)),
            'der': nn.Parameter(torch.tensor(1.0)),
            'combined': nn.Parameter(torch.tensor(1.0)),
            'meta': nn.Parameter(torch.tensor(1.0)),
        })

        # Initialize moving averages for branch losses
        self.moving_avg_loss = {
            'cli': None, 'der': None, 'combined': None, 'meta': None
        }
        self.alpha = 0.8  # Smoothing factor for moving average
        self.class_weights = class_weights
        self.epsilon = 1e-8  # Small value to avoid division by zero

    def update_moving_avg(self, loss, branch):
        #Update the moving average for the branch loss.
        if self.moving_avg_loss[branch] is None:
            # Initialize with first epoch's loss
            self.moving_avg_loss[branch] = loss.item()
        else:
            # Update moving average with smoothing
            self.moving_avg_loss[branch] = (
                self.alpha * self.moving_avg_loss[branch] + (1 - self.alpha) * loss.item()
            )

    def compute_branch_loss(self, cli_pred, der_pred, combined_pred, meta_pred, target, class_weight=None):
        # Compute individual branch losses
        loss_cli = F.cross_entropy(cli_pred, target, weight=class_weight.to(cli_pred.device) if class_weight is not None else None)
        loss_der = F.cross_entropy(der_pred, target, weight=class_weight.to(der_pred.device) if class_weight is not None else None)
        loss_combined = F.cross_entropy(combined_pred, target, weight=class_weight.to(combined_pred.device) if class_weight is not None else None)
        loss_meta = F.cross_entropy(meta_pred, target, weight=class_weight.to(meta_pred.device) if class_weight is not None else None)

        # Update moving averages
        self.update_moving_avg(loss_cli, 'cli')
        self.update_moving_avg(loss_der, 'der')
        self.update_moving_avg(loss_combined, 'combined')
        self.update_moving_avg(loss_meta, 'meta')

        # Calculate dynamic weights inversely proportional to moving average losses with stabilisation (epsilon)
        total_moving_avg = sum(self.moving_avg_loss.values()) + self.epsilon
        cli_weight = torch.clamp(torch.tensor((1 / (self.moving_avg_loss['cli'] + self.epsilon)) / total_moving_avg), 0.1, 10.0)
        der_weight = torch.clamp(torch.tensor((1 / (self.moving_avg_loss['der'] + self.epsilon)) / total_moving_avg), 0.1, 10.0)
        combined_weight = torch.clamp(torch.tensor((1 / (self.moving_avg_loss['combined'] + self.epsilon)) / total_moving_avg), 0.1, 10.0)
        meta_weight = torch.clamp(torch.tensor((1 / (self.moving_avg_loss['meta'] + self.epsilon)) / total_moving_avg), 0.1, 10.0)

        # Normalize weights to sum to 1 for stability
        weight_sum = cli_weight + der_weight + combined_weight + meta_weight
        cli_weight /= weight_sum
        der_weight /= weight_sum
        combined_weight /= weight_sum
        meta_weight /= weight_sum

        # Combine branch losses with dynamic weights
        total_loss = (
            cli_weight * loss_cli +
            der_weight * loss_der +
            combined_weight * loss_combined +
            meta_weight * loss_meta
        )

        return total_loss

    def forward(self, outputs, targets):
        # Use default class_weight if None
        if self.class_weights is None:
            diag_loss = self.compute_branch_loss(outputs['diag'][0], outputs['diag'][1], outputs['diag'][2], outputs['diag'][3], targets[0])
            pn_loss = self.compute_branch_loss(outputs['pn'][0], outputs['pn'][1], outputs['pn'][2], outputs['pn'][3], targets[1])
            bwv_loss = self.compute_branch_loss(outputs['bwv'][0], outputs['bwv'][1], outputs['bwv'][2], outputs['bwv'][3], targets[2])
            vs_loss = self.compute_branch_loss(outputs['vs'][0], outputs['vs'][1], outputs['vs'][2], outputs['vs'][3], targets[3])
            pig_loss = self.compute_branch_loss(outputs['pig'][0], outputs['pig'][1], outputs['pig'][2], outputs['pig'][3], targets[4])
            str_loss = self.compute_branch_loss(outputs['str'][0], outputs['str'][1], outputs['str'][2], outputs['str'][3], targets[5])
            dag_loss = self.compute_branch_loss(outputs['dag'][0], outputs['dag'][1], outputs['dag'][2], outputs['dag'][3], targets[6])
            rs_loss = self.compute_branch_loss(outputs['rs'][0], outputs['rs'][1], outputs['rs'][2], outputs['rs'][3], targets[7])
        else:
            diag_loss = self.compute_branch_loss(outputs['diag'][0], outputs['diag'][1], outputs['diag'][2], outputs['diag'][3], targets[0], self.class_weights[0])
            pn_loss = self.compute_branch_loss(outputs['pn'][0], outputs['pn'][1], outputs['pn'][2], outputs['pn'][3], targets[1], self.class_weights[1])
            bwv_loss = self.compute_branch_loss(outputs['bwv'][0], outputs['bwv'][1], outputs['bwv'][2], outputs['bwv'][3], targets[2], self.class_weights[2])
            vs_loss = self.compute_branch_loss(outputs['vs'][0], outputs['vs'][1], outputs['vs'][2], outputs['vs'][3], targets[3], self.class_weights[3])
            pig_loss = self.compute_branch_loss(outputs['pig'][0], outputs['pig'][1], outputs['pig'][2], outputs['pig'][3], targets[4], self.class_weights[4])
            str_loss = self.compute_branch_loss(outputs['str'][0], outputs['str'][1], outputs['str'][2], outputs['str'][3], targets[5], self.class_weights[5])
            dag_loss = self.compute_branch_loss(outputs['dag'][0], outputs['dag'][1], outputs['dag'][2], outputs['dag'][3], targets[6], self.class_weights[6])
            rs_loss = self.compute_branch_loss(outputs['rs'][0], outputs['rs'][1], outputs['rs'][2], outputs['rs'][3], targets[7], self.class_weights[7])

        # Final multi-task loss
        total_loss = (diag_loss + pn_loss + bwv_loss + vs_loss + pig_loss + str_loss + dag_loss + rs_loss) / 8

        return total_loss    
    
# Uncertainty aware voting    
def calculate_entropy(probs):
    #Calculate entropy of probability distribution.
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

def weighted_confidence_voting(cli_pred, der_pred, combined_pred, meta_pred):
    # Apply softmax to get probability distributions
    cli_probs = torch.softmax(cli_pred, dim=1)
    der_probs = torch.softmax(der_pred, dim=1)
    combined_probs = torch.softmax(combined_pred, dim=1)
    meta_probs = torch.softmax(meta_pred, dim=1)

    # Calculate entropy for each branch to assess uncertainty
    cli_entropy = calculate_entropy(cli_probs)
    der_entropy = calculate_entropy(der_probs)
    combined_entropy = calculate_entropy(combined_probs)
    meta_entropy = calculate_entropy(meta_probs)

    # Inverse of entropy for weights (1 - entropy normalized)
    total_entropy = cli_entropy + der_entropy + combined_entropy + meta_entropy
    cli_weight = (1 - cli_entropy / total_entropy).unsqueeze(1)
    der_weight = (1 - der_entropy / total_entropy).unsqueeze(1)
    combined_weight = (1 - combined_entropy / total_entropy).unsqueeze(1)
    meta_weight = (1 - meta_entropy / total_entropy).unsqueeze(1)

    # Weighted average of branch probabilities
    ensemble_probs = (
        cli_weight * cli_probs +
        der_weight * der_probs +
        combined_weight * combined_probs +
        meta_weight * meta_probs
    ) / (cli_weight + der_weight + combined_weight + meta_weight)

    # Final ensemble prediction
    ensemble_pred = torch.argmax(ensemble_probs, dim=1)

    return ensemble_probs, ensemble_pred
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super(CrossAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout_rate),
        )
        
    def forward(self, x1, x2):
        # Cross-attention from x1 -> x2
        x1 = x1 + self.attn(x1, x2, x2)[0]  # x1 attending to x2 features
        x1 = self.norm1(x1)
        
        # Feed-forward and residual connection
        x1 = x1 + self.mlp(x1)
        
        return x1
    
class DermFormer(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=256, model_name='jx_nest_tiny', num_heads=4, dropout_rate=0.5, meta_dim=32, device=device):
        super(DermFormer, self).__init__()
        # Load pretrained models for clinical, dermoscopy and context branches
        self.cli_nest = timm.create_model(model_name, pretrained=True)
        self.der_nest = timm.create_model(model_name, pretrained=True)
        self.context_nest = timm.create_model(model_name, pretrained=True)

        # Replace classifier heads with Identity layers to extract features
        num_features = self.cli_nest.get_classifier().in_features
        self.cli_nest.head = nn.Identity()
        self.der_nest.head = nn.Identity()
        self.context_nest.head = nn.Identity()

        # Input embedding layer for context branch
        self.context_embedding = nn.Sequential(
            nn.Conv2d(9, 3, kernel_size=1), 
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        self.meta_projection = nn.Linear(5, 3 * 224 * 224)  # Transform to match spatial dimensions (16, 3, 224, 224)
        
        self.meta_subnet = TabTransformer(
            categories=(3, 3, 9, 2, 3),
            num_continuous=0,
            dim=meta_dim,
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )
    
        # Cross-attention layers for cli and der with shared representation
        self.cli_cross_attention = CrossAttentionLayer(384, num_heads, dropout_rate)
        self.der_cross_attention = CrossAttentionLayer(384, num_heads, dropout_rate)
        #self.clider_cross_attention = CrossAttentionLayer(384, num_heads, dropout_rate)

        # Feature reduction layers
        self.cli_reduce = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.der_reduce = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        self.meta_reduce = nn.Sequential(
            nn.Linear(160, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )   

        # Feature reduction layer for concatenated features
        self.feature_reduce = nn.Sequential(
            nn.Linear(hidden_dim *3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.metacombined_reduce = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        

        # Classification heads
        self.fc_diag = nn.Linear(hidden_dim // 2, num_classes)
        self.fc_pn = nn.Linear(hidden_dim // 2, 3)
        self.fc_bwv = nn.Linear(hidden_dim // 2, 2)
        self.fc_vs = nn.Linear(hidden_dim // 2, 3)
        self.fc_pig = nn.Linear(hidden_dim // 2, 3)
        self.fc_str = nn.Linear(hidden_dim // 2, 3)
        self.fc_dag = nn.Linear(hidden_dim // 2, 3)
        self.fc_rs = nn.Linear(hidden_dim // 2, 2)
    
    def forward(self, meta_cat, meta_con, cli_x, der_x):
        #print(f"cli_x shape: {cli_x.shape}")
        #print(f"der_x shape: {der_x.shape}")
        #print(f"meta_cat shape: {meta_cat.shape}")
        # Process through both unimodal branches
        cli_h = self.cli_nest(cli_x)
        der_h = self.der_nest(der_x)
        #print(f"cli_features shape: {cli_h.shape}")
        #print(f"der_features shape: {der_h.shape}")
        
        # Reduce dimensionality of output features from cli and der branches
        cli_features = self.cli_reduce(cli_h)
        #print(f"cli_features shape: {cli_features.shape}")
        der_features = self.der_reduce(der_h)
        #print(f"der_features shape: {der_features.shape}")


        # Convert meta_cat to a Float from Long tensor
        meta_project = meta_cat.float()
        meta_projected = self.meta_projection(meta_project)  # Shape: [16, C]
        meta_projected = meta_projected.view(-1, 3, 224, 224)  # Shape: [16, 3, 224, 224]
        combined_x = torch.cat((cli_x, der_x, meta_projected), dim=1)  # Concatenate along channel dimension
        #print(f"combined_x shape: {combined_x.shape}")
        embedded_x = self.context_embedding(combined_x)
        #print(f"embedded_x shape: {embedded_x.shape}")
        context_h = self.context_nest(embedded_x)
        #print(f"context_h shape: {context_h.shape}")

        # Meta features through TabTransformer
        meta_features = self.meta_subnet(meta_cat, meta_con)
        #print(f"meta_features shape: {meta_features.shape}")
        meta_features = self.meta_reduce(meta_features)
        #print(f"meta_features shape: {meta_features.shape}")


        # Cross-attend cli and der with shared_h
        cli_context = self.cli_cross_attention(cli_h, context_h)
        #print(f"cli_context shape: {cli_context.shape}")
        der_context = self.der_cross_attention(der_h, context_h)
        #print(f"der_context shape: {der_context.shape}")

        #combined = self.clider_cross_attention(cli_context,der_context)
        #print(f"clider_context shape: {combined.shape}")
        # Combine Der_h with Cli_h features 
        combined = torch.cat((cli_context, der_context), dim=1)
        #print(f"combined shape: {combined.shape}")
        combined = self.feature_reduce(combined)
        #print(f"combined shape: {combined.shape}")

        # Combine meta_features with Der_h and cli_h features
        meta_combined = torch.cat((combined, meta_features), dim=1)
        meta_combined = self.metacombined_reduce(meta_combined)
        #print(f"meta_combined shape: {meta_combined.shape}")

        # Obtain logits from each branch separately
        diag_cli = self.fc_diag(cli_features)
        #print(f"diag_cli: {diag_cli.shape}")
        diag_der = self.fc_diag(der_features)
        #print(f"diag_der: {diag_der.shape}")

        diag_combined = self.fc_diag(combined)
        #print(f"der_combined: {diag_combined.shape}")

        diag_metacombined = self.fc_diag(meta_combined)
        #print(f"diag_meta: {diag_metacombined.shape}")
        
        # Confidence-based ensemble voting
        diag_max_confidence_probs, diag_max_confident_prediction = weighted_confidence_voting(diag_cli, diag_der,
                                                                                                diag_combined, 
                                                                                                diag_metacombined)

        # Repeat for other classification heads
        pn_cli = self.fc_pn(cli_features)
        pn_der = self.fc_pn(der_features)
        pn_combined = self.fc_pn(combined)
        pn_metacombined = self.fc_pn(meta_combined)
        # Confidence-based ensemble voting
        pn_max_confidence_probs, pn_max_confident_prediction = weighted_confidence_voting(pn_cli, pn_der, 
                                                                                            pn_combined, 
                                                                                            pn_metacombined)
        
        bwv_cli = self.fc_bwv(cli_features)
        bwv_der = self.fc_bwv(der_features)
        bwv_combined = self.fc_bwv(combined)
        bwv_metacombined = self.fc_bwv(meta_combined)
        # Confidence-based ensemble voting
        bwv_max_confidence_probs, bwv_max_confident_prediction = weighted_confidence_voting(bwv_cli, bwv_der, 
                                                                                            bwv_combined, 
                                                                                            bwv_metacombined)
        
        vs_cli = self.fc_vs(cli_features)
        vs_der = self.fc_vs(der_features)
        vs_combined = self.fc_vs(combined)
        vs_metacombined = self.fc_vs(meta_combined)
        # Confidence-based ensemble voting
        vs_max_confidence_probs, vs_max_confident_prediction = weighted_confidence_voting(vs_cli, vs_der, 
                                                                                            vs_combined, 
                                                                                            vs_metacombined)
        
        pig_cli = self.fc_pig(cli_features)
        pig_der = self.fc_pig(der_features)
        pig_combined = self.fc_pig(combined)
        pig_metacombined = self.fc_pig(meta_combined)
        # Confidence-based ensemble voting
        pig_max_confidence_probs, pig_max_confident_prediction = weighted_confidence_voting(pig_cli, pig_der, 
                                                                                            pig_combined, 
                                                                                            pig_metacombined)
        
        str_cli = self.fc_str(cli_features)
        str_der = self.fc_str(der_features)
        str_combined = self.fc_str(combined)
        str_metacombined = self.fc_str(meta_combined)
        # Confidence-based ensemble voting
        str_max_confidence_probs, str_max_confident_prediction = weighted_confidence_voting(str_cli, str_der, 
                                                                                            str_combined, 
                                                                                            str_metacombined)
        
        dag_cli = self.fc_dag(cli_features)
        dag_der = self.fc_dag(der_features)
        dag_combined = self.fc_dag(combined)
        dag_metacombined = self.fc_dag(meta_combined)
        # Confidence-based ensemble voting
        dag_max_confidence_probs, dag_max_confident_prediction = weighted_confidence_voting(dag_cli, dag_der, 
                                                                                            dag_combined, 
                                                                                            dag_metacombined)
        
        rs_cli = self.fc_rs(cli_features)
        rs_der = self.fc_rs(der_features)
        rs_combined = self.fc_rs(combined)
        rs_metacombined = self.fc_rs(meta_combined)
        # Confidence-based ensemble voting
        rs_max_confidence_probs, rs_max_confident_prediction = weighted_confidence_voting(rs_cli, rs_der, 
                                                                                            rs_combined, 
                                                                                            rs_metacombined)
        # Return predictions for each branch and ensemble, organized by task
        return {
            'diag': [diag_cli, diag_der, diag_combined, diag_metacombined, diag_max_confidence_probs, diag_max_confident_prediction],
            'pn': [pn_cli, pn_der, pn_combined, pn_metacombined, pn_max_confidence_probs, pn_max_confident_prediction],
            'bwv': [bwv_cli, bwv_der, bwv_combined, bwv_metacombined, bwv_max_confidence_probs, bwv_max_confident_prediction],
            'vs': [vs_cli, vs_der, vs_combined, vs_combined, vs_metacombined, vs_max_confidence_probs, vs_max_confident_prediction],
            'pig': [pig_cli, pig_der, pig_combined, pig_combined, pig_metacombined, pig_max_confidence_probs, pig_max_confident_prediction],
            'str': [str_cli, str_der, str_combined, str_combined, str_metacombined, str_max_confidence_probs, str_max_confident_prediction],
            'dag': [dag_cli, dag_der, dag_combined, dag_combined, dag_metacombined, dag_max_confidence_probs, dag_max_confident_prediction],
            'rs': [rs_cli, rs_der, rs_combined, rs_combined, rs_metacombined, rs_max_confidence_probs, rs_max_confident_prediction]
        }

def main():
    #torch.cuda.empty_cache()
    model_names_with_pretrained = timm.list_models('*nest*', pretrained=True)
    #print("Available models with *nest* and pretrained:", model_names_with_pretrained)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the MM_nest model instance
    model = DermFormer(num_classes=5, hidden_dim=256).to(device)
    #print("Model architecture:\n", model)
    
    # Create dummy inputs
    cli_x = torch.randn(16, 3, 224, 224).to(device)  # Clinical images
    der_x = torch.randn(16, 3, 224, 224).to(device)  # Dermoscopy images
    meta_cat = torch.cat([
        torch.randint(0, 3, (16, 1)),  
        torch.randint(0, 3, (16, 1)),
        torch.randint(0, 9, (16, 1)),  
        torch.randint(0, 2, (16, 1)),  
        torch.randint(0, 3, (16, 1))
    ], dim=1).to(device)
    
    meta_con = torch.empty((16, 0)).to(device)  # No continuous metadata

    # Forward pass
    outputs = model(meta_cat, meta_con, cli_x, der_x)
    """
    print(f"Outputs type: {type(outputs)}")

    # Print the outputs for each task
    for task_name, task_outputs in outputs.items():
        print(f"\nTask: {task_name}")
        for i, output in enumerate(task_outputs):
            print(f"  Branch {i} output shape: {output.shape}")
    """
if __name__ == "__main__":
    main()