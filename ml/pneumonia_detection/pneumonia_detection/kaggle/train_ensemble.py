"""
PNEUMONIA DETECTION - ENSEMBLE TRAINING (KAGGLE)
=================================================
Run this on Kaggle with GPU (T4/P100) enabled.
Add dataset: rsna-pneumonia-detection-challenge

INSTRUCTIONS:
1. Create a new Kaggle notebook
2. Add the RSNA Pneumonia Detection Challenge dataset
3. Enable GPU accelerator
4. Copy-paste this entire script into cells (split at ### CELL markers)
5. Run all cells
6. Download the /kaggle/working/output/ folder when done
"""

### CELL 1: SETUP
import os, json, time, warnings
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, f1_score
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

warnings.filterwarnings('ignore')

CONFIG = {
    'data_dir': '/kaggle/input/rsna-pneumonia-detection-challenge',
    'output_dir': '/kaggle/working/output',
    'processed_dir': '/kaggle/working/processed',
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 2,
    'seed': 42,
    'phase1_epochs': 5,
    'phase2_epochs': 15,
    'phase1_lr': 1e-3,
    'phase2_lr_head': 1e-3,
    'phase2_lr_base': 1e-5,
    'patience': 7,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

for d in [CONFIG['output_dir'], f"{CONFIG['output_dir']}/figures", CONFIG['processed_dir']]:
    os.makedirs(d, exist_ok=True)

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
print(f"Device: {CONFIG['device']}")

### CELL 2: LOAD LABELS
labels_df = pd.read_csv(f"{CONFIG['data_dir']}/stage_2_train_labels.csv")

# Handle 3-class issue: group by patient, max Target
# Target=0 includes both "Normal" and "No Lung Opacity / Not Normal"
# Target=1 is "Lung Opacity" (pneumonia)
patient_labels = labels_df.groupby('patientId')['Target'].max().reset_index()
patient_labels.columns = ['patientId', 'label']

# Check for detailed class info
detailed_path = f"{CONFIG['data_dir']}/stage_2_detailed_class_info.csv"
if os.path.exists(detailed_path):
    detailed_df = pd.read_csv(detailed_path)
    print("Detailed class distribution:")
    print(detailed_df['class'].value_counts())
    print()

n_normal = (patient_labels['label'] == 0).sum()
n_pneumonia = (patient_labels['label'] == 1).sum()
print(f"Binary labels -> Normal: {n_normal}, Pneumonia: {n_pneumonia}")
print(f"Imbalance ratio: {n_normal/n_pneumonia:.2f}:1")

### CELL 3: TRAIN/VAL/TEST SPLIT
train_df, temp_df = train_test_split(
    patient_labels, test_size=0.3, random_state=CONFIG['seed'],
    stratify=patient_labels['label'])
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=CONFIG['seed'],
    stratify=temp_df['label'])

print(f"Train: {len(train_df)} (pos: {train_df['label'].sum()})")
print(f"Val:   {len(val_df)} (pos: {val_df['label'].sum()})")
print(f"Test:  {len(test_df)} (pos: {test_df['label'].sum()})")

split_info = {
    'train': int(len(train_df)), 'val': int(len(val_df)), 'test': int(len(test_df)),
    'train_pos': int(train_df['label'].sum()),
    'val_pos': int(val_df['label'].sum()),
    'test_pos': int(test_df['label'].sum())
}
with open(f"{CONFIG['output_dir']}/split_info.json", 'w') as f:
    json.dump(split_info, f, indent=2)

### CELL 4: PREPROCESS DICOMS TO NPY
def preprocess_dicom(patient_id, data_dir, size=224):
    """Read DICOM, extract pixels, resize to size x size, normalize to [0,1]."""
    path = os.path.join(data_dir, 'stage_2_train_images', f'{patient_id}.dcm')
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    if img.max() > 0:
        img = img / img.max()
    pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil).astype(np.float32) / 255.0

print("Preprocessing DICOMs to .npy ...")
all_pids = patient_labels['patientId'].tolist()
for i, pid in enumerate(all_pids):
    out = os.path.join(CONFIG['processed_dir'], f'{pid}.npy')
    if not os.path.exists(out):
        try:
            np.save(out, preprocess_dicom(pid, CONFIG['data_dir'], CONFIG['image_size']))
        except Exception as e:
            print(f"  Error {pid}: {e}")
    if (i + 1) % 5000 == 0:
        print(f"  {i+1}/{len(all_pids)}")
print("Done preprocessing.")

# Compute norm stats from training set (sample 5000 for speed)
print("Computing normalization stats...")
sample_pids = train_df['patientId'].tolist()[:5000]
pixels = []
for pid in sample_pids:
    p = os.path.join(CONFIG['processed_dir'], f'{pid}.npy')
    if os.path.exists(p):
        pixels.append(np.load(p).flatten())
pixels = np.concatenate(pixels)
norm_stats = {'mean': float(np.mean(pixels)), 'std': float(np.std(pixels))}
with open(f"{CONFIG['output_dir']}/norm_stats.json", 'w') as f:
    json.dump(norm_stats, f, indent=2)
print(f"Mean: {norm_stats['mean']:.4f}, Std: {norm_stats['std']:.4f}")
del pixels

### CELL 5: DATASET CLASS
import torchvision.transforms as T

class CXRDataset(Dataset):
    def __init__(self, df, proc_dir, norm, augment=False):
        self.df = df.reset_index(drop=True)
        self.proc_dir = proc_dir
        self.mean, self.std = norm['mean'], norm['std']
        self.augment = augment
        self.aug = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(10),
            T.RandomAffine(0, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.load(os.path.join(self.proc_dir, f"{row['patientId']}.npy"))
        pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
        if self.augment:
            pil = self.aug(pil)
        img = np.array(pil).astype(np.float32) / 255.0
        img = (img - self.mean) / (self.std + 1e-8)
        img3 = np.stack([img, img, img], axis=0)
        return torch.FloatTensor(img3), int(row['label'])

### CELL 6: DATALOADERS
train_ds = CXRDataset(train_df, CONFIG['processed_dir'], norm_stats, augment=True)
val_ds = CXRDataset(val_df, CONFIG['processed_dir'], norm_stats)
test_ds = CXRDataset(test_df, CONFIG['processed_dir'], norm_stats)

# Weighted sampler
weights = 1.0 / np.bincount(train_df['label'].values).astype(np.float64)
sample_w = weights[train_df['label'].values]
sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                          sampler=sampler, num_workers=CONFIG['num_workers'])
val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                        shuffle=False, num_workers=CONFIG['num_workers'])
test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'],
                         shuffle=False, num_workers=CONFIG['num_workers'])

x, y = next(iter(train_loader))
print(f"Batch: {x.shape}, Labels dist: {torch.bincount(torch.tensor(y))}")

### CELL 7: MODEL BUILDERS
def build_densenet121():
    m = models.densenet121(weights='IMAGENET1K_V1')
    nf = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Linear(nf, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 2))
    return m

def build_efficientnet_b0():
    m = models.efficientnet_b0(weights='IMAGENET1K_V1')
    nf = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(nf, 512), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(512, 2))
    return m

def freeze_backbone(m, name):
    target = m.features if name in ('densenet121', 'efficientnet_b0') else None
    if target:
        for p in target.parameters():
            p.requires_grad = False

def unfreeze_last(m, name):
    if name == 'densenet121':
        for n, p in m.features.named_parameters():
            if any(k in n for k in ['denseblock4', 'transition3', 'norm5']):
                p.requires_grad = True
    elif name == 'efficientnet_b0':
        for n, p in m.features.named_parameters():
            if any(f'.{i}.' in n for i in [6, 7, 8]):
                p.requires_grad = True

### CELL 8: EVALUATION FUNCTION
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), torch.tensor(y).to(device)
        out = model(x)
        loss_sum += criterion(out, y).item()
        probs = torch.softmax(out, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    avg_loss = loss_sum / len(loader)
    acc = 100.0 * correct / total
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auc, np.array(all_probs), np.array(all_labels)

### CELL 9: TRAINING FUNCTION
def train_model(model, model_name, train_loader, val_loader, config):
    device = config['device']
    model = model.to(device)
    out_dir = config['output_dir']

    # Class-weighted loss
    pos_w = (train_df['label'] == 0).sum() / max((train_df['label'] == 1).sum(), 1)
    cw = torch.FloatTensor([1.0, pos_w]).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    metrics = []
    best_auc = 0.0
    patience_ctr = 0

    # === PHASE 1: Frozen backbone ===
    print(f"\n{'='*50}")
    print(f"[{model_name}] PHASE 1: Head only ({config['phase1_epochs']} epochs)")
    print(f"{'='*50}")
    freeze_backbone(model, model_name)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=config['phase1_lr'])

    for ep in range(config['phase1_epochs']):
        model.train()
        rl, cor, tot = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), torch.tensor(yb).to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            rl += loss.item()
            _, pred = out.max(1)
            cor += pred.eq(yb).sum().item()
            tot += yb.size(0)
        tl, ta = rl / len(train_loader), 100.0 * cor / tot
        vl, va, vauc, _, _ = evaluate(model, val_loader, criterion, device)
        metrics.append({'epoch': ep+1, 'phase': 1, 'train_loss': tl,
                       'train_acc': ta, 'val_loss': vl, 'val_acc': va, 'val_auc': vauc})
        print(f"  Ep {ep+1}: TrL={tl:.4f} TrA={ta:.1f}% | VL={vl:.4f} VA={va:.1f}% AUC={vauc:.4f}")
        if vauc > best_auc:
            best_auc = vauc
            torch.save(model.state_dict(), f"{out_dir}/{model_name}_best.pth")
            patience_ctr = 0
        else:
            patience_ctr += 1

    # === PHASE 2: Unfreeze last blocks ===
    print(f"\n{'='*50}")
    print(f"[{model_name}] PHASE 2: Fine-tune ({config['phase2_epochs']} epochs)")
    print(f"{'='*50}")
    unfreeze_last(model, model_name)

    if model_name == 'densenet121':
        base_params = [p for n, p in model.features.named_parameters()
                       if p.requires_grad and any(k in n for k in ['denseblock4','transition3','norm5'])]
        head_params = list(model.classifier.parameters())
    else:
        base_params = [p for n, p in model.features.named_parameters() if p.requires_grad]
        head_params = list(model.classifier.parameters())

    opt = optim.Adam([
        {'params': base_params, 'lr': config['phase2_lr_base']},
        {'params': head_params, 'lr': config['phase2_lr_head']}
    ])
    sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)
    patience_ctr = 0

    for ep in range(config['phase2_epochs']):
        model.train()
        rl, cor, tot = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), torch.tensor(yb).to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            rl += loss.item()
            _, pred = out.max(1)
            cor += pred.eq(yb).sum().item()
            tot += yb.size(0)
        sched.step()
        tl, ta = rl / len(train_loader), 100.0 * cor / tot
        vl, va, vauc, _, _ = evaluate(model, val_loader, criterion, device)
        total_ep = config['phase1_epochs'] + ep + 1
        metrics.append({'epoch': total_ep, 'phase': 2, 'train_loss': tl,
                       'train_acc': ta, 'val_loss': vl, 'val_acc': va, 'val_auc': vauc})
        print(f"  Ep {total_ep}: TrL={tl:.4f} TrA={ta:.1f}% | VL={vl:.4f} VA={va:.1f}% AUC={vauc:.4f}")
        if vauc > best_auc:
            best_auc = vauc
            torch.save(model.state_dict(), f"{out_dir}/{model_name}_best.pth")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config['patience']:
                print(f"  Early stopping at epoch {total_ep}")
                break

    print(f"\n[{model_name}] Best Val AUC: {best_auc:.4f}")
    # Save metrics
    pd.DataFrame(metrics).to_csv(f"{out_dir}/{model_name}_metrics.csv", index=False)
    # Reload best weights
    model.load_state_dict(torch.load(f"{out_dir}/{model_name}_best.pth"))
    return model, metrics

### CELL 10: TRAIN DENSENET121
print("\n" + "="*60)
print("TRAINING DENSENET121")
print("="*60)
densenet = build_densenet121()
densenet, dn_metrics = train_model(densenet, 'densenet121', train_loader, val_loader, CONFIG)

### CELL 11: TRAIN EFFICIENTNET-B0
print("\n" + "="*60)
print("TRAINING EFFICIENTNET-B0")
print("="*60)
effnet = build_efficientnet_b0()
effnet, en_metrics = train_model(effnet, 'efficientnet_b0', train_loader, val_loader, CONFIG)

### CELL 12: EVALUATE INDIVIDUAL + ENSEMBLE ON TEST SET
device = CONFIG['device']
pos_w = (train_df['label'] == 0).sum() / max((train_df['label'] == 1).sum(), 1)
cw = torch.FloatTensor([1.0, pos_w]).to(device)
criterion = nn.CrossEntropyLoss(weight=cw)

# Individual evaluations
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

dn_loss, dn_acc, dn_auc, dn_probs, test_labels = evaluate(
    densenet, test_loader, criterion, device)
print(f"DenseNet121  -> Acc: {dn_acc:.2f}% | AUC: {dn_auc:.4f}")

en_loss, en_acc, en_auc, en_probs, _ = evaluate(
    effnet, test_loader, criterion, device)
print(f"EfficientNet -> Acc: {en_acc:.2f}% | AUC: {en_auc:.4f}")

# Ensemble (soft voting)
ens_probs = (dn_probs + en_probs) / 2.0
ens_preds = (ens_probs >= 0.5).astype(int)
ens_acc = 100.0 * np.mean(ens_preds == test_labels)
ens_auc = roc_auc_score(test_labels, ens_probs)
print(f"ENSEMBLE     -> Acc: {ens_acc:.2f}% | AUC: {ens_auc:.4f}")

# Classification report
print("\n--- Ensemble Classification Report ---")
report = classification_report(test_labels, ens_preds,
                               target_names=['Normal', 'Pneumonia'], output_dict=True)
print(classification_report(test_labels, ens_preds, target_names=['Normal', 'Pneumonia']))

# Save test metrics
test_metrics = {
    'densenet121': {'accuracy': dn_acc, 'auc': dn_auc},
    'efficientnet_b0': {'accuracy': en_acc, 'auc': en_auc},
    'ensemble': {
        'accuracy': ens_acc, 'auc': ens_auc,
        'precision_normal': report['Normal']['precision'],
        'recall_normal': report['Normal']['recall'],
        'f1_normal': report['Normal']['f1-score'],
        'precision_pneumonia': report['Pneumonia']['precision'],
        'recall_pneumonia': report['Pneumonia']['recall'],
        'f1_pneumonia': report['Pneumonia']['f1-score'],
        'macro_f1': report['macro avg']['f1-score'],
    }
}
with open(f"{CONFIG['output_dir']}/test_metrics.json", 'w') as f:
    json.dump(test_metrics, f, indent=2)

### CELL 13: GENERATE FIGURES
fig_dir = f"{CONFIG['output_dir']}/figures"

# 1. Confusion Matrix
cm = confusion_matrix(test_labels, ens_preds)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Normal', 'Pneumonia'])
ax.set_yticklabels(['Normal', 'Pneumonia'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Ensemble Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f"{fig_dir}/confusion_matrix.png", dpi=150)
plt.close()

# 2. ROC Curve
fig, ax = plt.subplots(figsize=(7, 6))
for name, probs in [('DenseNet121', dn_probs), ('EfficientNet-B0', en_probs), ('Ensemble', ens_probs)]:
    fpr, tpr, _ = roc_curve(test_labels, probs)
    auc_val = roc_auc_score(test_labels, probs)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.4f})')
ax.plot([0,1], [0,1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves'); ax.legend(); plt.tight_layout()
plt.savefig(f"{fig_dir}/roc_curves.png", dpi=150)
plt.close()

# 3. Precision-Recall Curve
fig, ax = plt.subplots(figsize=(7, 6))
for name, probs in [('DenseNet121', dn_probs), ('EfficientNet-B0', en_probs), ('Ensemble', ens_probs)]:
    prec, rec, _ = precision_recall_curve(test_labels, probs)
    ax.plot(rec, prec, label=name)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves'); ax.legend(); plt.tight_layout()
plt.savefig(f"{fig_dir}/pr_curves.png", dpi=150)
plt.close()

# 4. Training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for name, mfile in [('DenseNet121', 'densenet121_metrics.csv'),
                     ('EfficientNet-B0', 'efficientnet_b0_metrics.csv')]:
    df = pd.read_csv(f"{CONFIG['output_dir']}/{mfile}")
    axes[0].plot(df['epoch'], df['train_loss'], label=f'{name} Train')
    axes[0].plot(df['epoch'], df['val_loss'], '--', label=f'{name} Val')
    axes[1].plot(df['epoch'], df['train_acc'], label=f'{name} Train')
    axes[1].plot(df['epoch'], df['val_acc'], '--', label=f'{name} Val')
    axes[2].plot(df['epoch'], df['val_auc'], label=name)

axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()
axes[1].set_title('Accuracy (%)'); axes[1].set_xlabel('Epoch'); axes[1].legend()
axes[2].set_title('Val AUC'); axes[2].set_xlabel('Epoch'); axes[2].legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/training_curves.png", dpi=150)
plt.close()

# Merge metrics into one CSV
dn_df = pd.read_csv(f"{CONFIG['output_dir']}/densenet121_metrics.csv")
en_df = pd.read_csv(f"{CONFIG['output_dir']}/efficientnet_b0_metrics.csv")
dn_df['model'] = 'densenet121'
en_df['model'] = 'efficientnet_b0'
all_metrics = pd.concat([dn_df, en_df], ignore_index=True)
all_metrics.to_csv(f"{CONFIG['output_dir']}/training_metrics.csv", index=False)

print("\n" + "="*60)
print("ALL DONE! Download the /kaggle/working/output/ folder.")
print("="*60)
print(f"\nFiles saved:")
for f in os.listdir(CONFIG['output_dir']):
    fp = os.path.join(CONFIG['output_dir'], f)
    if os.path.isfile(fp):
        print(f"  {f} ({os.path.getsize(fp)/1024:.1f} KB)")
    elif os.path.isdir(fp):
        for ff in os.listdir(fp):
            print(f"  {f}/{ff}")
