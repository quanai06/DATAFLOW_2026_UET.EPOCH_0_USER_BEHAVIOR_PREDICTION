import torch
import torch.nn as nn


class TransformerModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        num_classes_list,   # list 6 numbers
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_ff=256,
        dropout=0.2,
        max_len=100
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ===== multi heads =====
        # combined_dim = d_model * 3 #neu them last/first token
        combined_dim = d_model
        self.heads = nn.ModuleList([
            nn.Linear(combined_dim, n) for n in num_classes_list
        ])

    def forward(self, x, mask):

        batch_size, seq_len = x.shape
    
        pos = torch.arange(seq_len, device=x.device)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)
    
        x = self.embedding(x) + self.pos_embedding(pos)
    
        x = self.encoder(x, src_key_padding_mask=mask)
    
        # # ===== first token =====
        # first_token_emb = x[:, 0, :]
    
        # ===== masked mean pooling =====
        mask_inv = (~mask).unsqueeze(-1)
        x_masked = x * mask_inv
    
        sum_x = x_masked.sum(dim=1)
        count = mask_inv.sum(dim=1).clamp(min=1)
        pooled = sum_x / count
    
        # ===== last valid token =====
        lengths = mask_inv.squeeze(-1).sum(dim=1).long()
        last_indices = (lengths - 1).clamp(min=0)
    
        # last_token_emb = x[torch.arange(batch_size, device=x.device), last_indices]
    
        # ===== concat =====
        # combined = torch.cat([pooled, first_token_emb, last_token_emb], dim=1)
        combined = torch.cat([pooled], dim=1)

    
        outputs = [head(combined) for head in self.heads]
    
        return outputs