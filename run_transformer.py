import sys, pickle, os, torch
sys.path.insert(0, '.')
from src.training.train_transformer import train_transformer, train_full_model
from src.data.loaders import get_transformer_loaders
from src.models.transformer_model import TransformerModel
from src.metrics.metrics import evaluate_model

with open('data/layer3_features/transformer/action_remapper.pkl', 'rb') as f:
    remap_data = pickle.load(f)

config = {
    'feature_path': 'data/layer3_features/transformer',
    'batch_size': 64,
    'epochs': 20,
    'patience': 4,
    'vocab_size': remap_data['vocab_size'],
    'num_classes_list': [12, 31, 99, 12, 31, 99],
    'd_model': 256,
    'nhead': 8,
    'num_layers': 4,
    'dim_ff': 512,
    'max_len': 37,
    'n_splits': 5,
}
print(f"vocab_size: {config['vocab_size']}", flush=True)

print("=== KFold training ===", flush=True)
train_transformer(config)

print("=== Full model training ===", flush=True)
train_full_model(config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, val_loader = get_transformer_loaders(config)

models = []
for fold in range(5):
    ckpt = torch.load(f'models/transformer/transformer_fold_{fold}.pt', map_location=device)
    nsf = ckpt.get('num_stat_features', 0)
    m = TransformerModel(
        vocab_size=config['vocab_size'],
        num_classes_list=config['num_classes_list'],
        d_model=256, nhead=8, num_layers=4, dim_ff=512, max_len=37,
        num_stat_features=nsf,
    )
    m.load_state_dict(ckpt['model_state_dict'])
    m.to(device).eval()
    models.append(m)

print("=== ENSEMBLE VAL RESULTS ===", flush=True)
val_res = evaluate_model(models, val_loader, device)
print(f"Exact Match: {val_res['exact_match_accuracy']:.4f}", flush=True)
print(f"Macro F1:    {val_res['macro_f1_score']:.4f}", flush=True)
for i, f1 in enumerate(val_res['f1_per_attribute']):
    print(f"  attr_{i+1}: {f1:.4f}", flush=True)
