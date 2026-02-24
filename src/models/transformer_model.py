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
        self.heads = nn.ModuleList([
            nn.Linear(d_model, n) for n in num_classes_list
        ])

    def forward(self, x, mask):

        batch_size, seq_len = x.shape

        pos = torch.arange(seq_len, device=x.device)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)

        x = self.embedding(x) + self.pos_embedding(pos)

        x = self.encoder(x, src_key_padding_mask=mask)

        # masked mean pooling
        mask_inv = (~mask).unsqueeze(-1)
        x = x * mask_inv

        sum_x = x.sum(dim=1)
        count = mask_inv.sum(dim=1).clamp(min=1)

        pooled = sum_x / count

        outputs = [head(pooled) for head in self.heads]

        return outputs