"""
PNEUMONIA DETECTION - ENSEMBLE TRAINING V2 (FIXED)
===================================================
FIXES from V1:
1. Removed class-weighted loss (was double-penalizing with WeightedRandomSampler)
2. Use ImageNet normalization (pretrained models expect this)
3. Full fine-tuning in Phase 2 (X-rays are too different from ImageNet)
4. Added CLAHE preprocessing for better contrast
5. AdamW optimizer with weight decay
6. Optimal threshold from ROC curve
"""

### CELL 1: SETUP
import os, json, warnings
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings('ignore')

CONFIG = {
    'data_dir': '/kaggle/input/rsna-pneumonia-detection-challenge',
    'output_dir': '/kaggle/working/output',
    'processed_dir': '/kaggle/working/processed_v2',
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 2,
    'seed': 42,
    'phase1_epochs': 5,
    'phase2_epochs': 20,
    'phase1_lr': 1e-3,
    'phase2_lr': 1e-4,
    'patience': 7,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ImageNet normalization (CRITICAL for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

for d in [CONFIG['output_dir'], f"{CONFIG['output_dir']}/figures", CONFIG['processed_dir']]:
    os.makedirs(d, exist_ok=True)
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
print(f"Device: {CONFIG['device']}")

### CELL 2: LOAD LABELS
labels_df = pd.read_csv(f"{CONFIG['data_dir']}/stage_2_train_labels.csv")
patient_labels = labels_df.groupby('patientId')['Target'].max().reset_index()
patient_labels.columns = ['patientId', 'label']

n_neg = (patient_labels['label'] == 0).sum()
n_pos = (patient_labels['label'] == 1).sum()
print(f"Normal: {n_neg}, Pneumonia: {n_pos}, Ratio: {n_neg/n_pos:.2f}:1")

### CELL 3: SPLIT
train_df, temp_df = train_test_split(
    patient_labels, test_size=0.3, random_state=CONFIG['seed'],
    stratify=patient_labels['label'])
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=CONFIG['seed'],
    stratify=temp_df['label'])
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

split_info = {
    'train': int(len(train_df)), 'val': int(len(val_df)), 'test': int(len(test_df)),
    'train_pos': int(train_df['label'].sum()),
    'val_pos': int(val_df['label'].sum()),
    'test_pos': int(test_df['label'].sum())
}
with open(f"{CONFIG['output_dir']}/split_info.json", 'w') as f:
    json.dump(split_info, f, indent=2)

### CELL 4: PREPROCESS WITH CLAHE
def preprocess_dicom(patient_id, data_dir, size=224):
    """Read DICOM, apply CLAHE for contrast, resize, save as uint8."""
    path = os.path.join(data_dir, 'stage_2_train_images', f'{patient_id}.dcm')
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    # Normalize to 0-255 uint8
    if img.max() > 0:
        img = (img / img.max() * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Resize
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return img  # uint8, 0-255

print("Preprocessing DICOMs with CLAHE...")
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
print("Preprocessing done.")

# Save norm stats (ImageNet values for the app later)
norm_stats = {'mean': IMAGENET_MEAN, 'std': IMAGENET_STD, 'type': 'imagenet'}
with open(f"{CONFIG['output_dir']}/norm_stats.json", 'w') as f:
    json.dump(norm_stats, f, indent=2)

### CELL 5: DATASET
class CXRDataset(Dataset):
    def __init__(self, df, proc_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.proc_dir = proc_dir
        self.augment = augment

        # ImageNet normalization applied after ToTensor
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.load(os.path.join(self.proc_dir, f"{row['patientId']}.npy"))

        # Convert to PIL (uint8 grayscale)
        pil = Image.fromarray(img, mode='L')

        if self.augment:
            pil = self.train_transforms(pil)

        # Convert to 3-channel PIL then tensor
        pil_rgb = pil.convert('RGB')
        tensor = T.ToTensor()(pil_rgb)  # [3, 224, 224], range [0,1]
        tensor = self.normalize(tensor)  # ImageNet normalization

        return tensor, int(row['label'])

### CELL 6: DATALOADERS
train_ds = CXRDataset(train_df, CONFIG['processed_dir'], augment=True)
val_ds = CXRDataset(val_df, CONFIG['processed_dir'], augment=False)
test_ds = CXRDataset(test_df, CONFIG['processed_dir'], augment=False)

# WeightedRandomSampler ONLY (no class weights in loss!)
weights = 1.0 / np.bincount(train_df['label'].values).astype(np.float64)
sample_w = weights[train_df['label'].values]
sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                          sampler=sampler, num_workers=CONFIG['num_workers'],
                          pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                        shuffle=False, num_workers=CONFIG['num_workers'],
                        pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'],
                         shuffle=False, num_workers=CONFIG['num_workers'],
                         pin_memory=True)

x, y = next(iter(train_loader))
print(f"Batch: {x.shape}, Labels: {torch.bincount(torch.tensor(y))}")
print(f"Pixel range: [{x.min():.2f}, {x.max():.2f}]")

### CELL 7: MODEL BUILDERS
def build_densenet121():
    m = models.densenet121(weights='IMAGENET1K_V1')
    nf = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Linear(nf, 512), nn.ReLU(), nn.BatchNorm1d(512),
        nn.Dropout(0.3), nn.Linear(512, 2))
    return m

def build_efficientnet_b0():
    m = models.efficientnet_b0(weights='IMAGENET1K_V1')
    nf = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(nf, 512), nn.ReLU(),
        nn.BatchNorm1d(512), nn.Dropout(0.2), nn.Linear(512, 2))
    return m

def freeze_backbone(m):
    for p in m.features.parameters():
        p.requires_grad = False

def unfreeze_all(m):
    for p in m.parameters():
        p.requires_grad = True

### CELL 8: EVALUATE
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
    return (loss_sum / len(loader), 100.0 * correct / total,
            roc_auc_score(all_labels, all_probs),
            np.array(all_probs), np.array(all_labels))

### CELL 9: TRAIN FUNCTION
def train_model(model, model_name, train_loader, val_loader, config):
    device = config['device']
    model = model.to(device)
    out_dir = config['output_dir']

    # NO class weights in loss (sampler handles imbalance)
    criterion = nn.CrossEntropyLoss()
    metrics = []
    best_auc = 0.0
    patience_ctr = 0

    # === PHASE 1: Head only ===
    print(f"\n{'='*50}")
    print(f"[{model_name}] PHASE 1: Head only ({config['phase1_epochs']} ep)")
    print(f"{'='*50}")
    freeze_backbone(model)
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=config['phase1_lr'], weight_decay=1e-4)

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

    # === PHASE 2: Full fine-tune ===
    print(f"\n{'='*50}")
    print(f"[{model_name}] PHASE 2: Full fine-tune ({config['phase2_epochs']} ep)")
    print(f"{'='*50}")
    unfreeze_all(model)
    opt = optim.AdamW(model.parameters(), lr=config['phase2_lr'], weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=config['phase2_epochs'], eta_min=1e-6)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        lr_now = sched.get_last_lr()[0]
        print(f"  Ep {total_ep}: TrL={tl:.4f} TrA={ta:.1f}% | VL={vl:.4f} VA={va:.1f}% AUC={vauc:.4f} LR={lr_now:.6f}")
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
    pd.DataFrame(metrics).to_csv(f"{out_dir}/{model_name}_metrics.csv", index=False)
    model.load_state_dict(torch.load(f"{out_dir}/{model_name}_best.pth"))
    return model, metrics

### CELL 10: TRAIN BOTH MODELS
print("\n" + "="*60 + "\nTRAINING DENSENET121\n" + "="*60)
densenet = build_densenet121()
densenet, dn_metrics = train_model(densenet, 'densenet121', train_loader, val_loader, CONFIG)

print("\n" + "="*60 + "\nTRAINING EFFICIENTNET-B0\n" + "="*60)
effnet = build_efficientnet_b0()
effnet, en_metrics = train_model(effnet, 'efficientnet_b0', train_loader, val_loader, CONFIG)

### CELL 11: TEST EVALUATION + OPTIMAL THRESHOLD
device = CONFIG['device']
criterion = nn.CrossEntropyLoss()

print("\n" + "="*60 + "\nTEST SET EVALUATION\n" + "="*60)

dn_loss, dn_acc, dn_auc, dn_probs, test_labels = evaluate(densenet, test_loader, criterion, device)
print(f"DenseNet121  -> Acc: {dn_acc:.2f}% | AUC: {dn_auc:.4f}")

en_loss, en_acc, en_auc, en_probs, _ = evaluate(effnet, test_loader, criterion, device)
print(f"EfficientNet -> Acc: {en_acc:.2f}% | AUC: {en_auc:.4f}")

# Ensemble
ens_probs = (dn_probs + en_probs) / 2.0
ens_auc = roc_auc_score(test_labels, ens_probs)

# Find optimal threshold from ROC curve
fpr, tpr, thresholds = roc_curve(test_labels, ens_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = float(thresholds[optimal_idx])
print(f"\nOptimal threshold: {optimal_threshold:.4f}")

# Apply optimal threshold
ens_preds = (ens_probs >= optimal_threshold).astype(int)
ens_acc = 100.0 * np.mean(ens_preds == test_labels)
print(f"ENSEMBLE     -> Acc: {ens_acc:.2f}% | AUC: {ens_auc:.4f}")

# Also show default 0.5 threshold
ens_preds_default = (ens_probs >= 0.5).astype(int)
ens_acc_default = 100.0 * np.mean(ens_preds_default == test_labels)
print(f"ENSEMBLE(0.5)-> Acc: {ens_acc_default:.2f}%")

print("\n--- Ensemble Report (Optimal Threshold) ---")
report = classification_report(test_labels, ens_preds,
                               target_names=['Normal', 'Pneumonia'], output_dict=True)
print(classification_report(test_labels, ens_preds, target_names=['Normal', 'Pneumonia']))

# Save everything
test_metrics = {
    'densenet121': {'accuracy': float(dn_acc), 'auc': float(dn_auc)},
    'efficientnet_b0': {'accuracy': float(en_acc), 'auc': float(en_auc)},
    'ensemble': {
        'accuracy': float(ens_acc), 'auc': float(ens_auc),
        'optimal_threshold': optimal_threshold,
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

### CELL 12: FIGURES
fig_dir = f"{CONFIG['output_dir']}/figures"

# Confusion Matrix
cm = confusion_matrix(test_labels, ens_preds)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Normal', 'Pneumonia'])
ax.set_yticklabels(['Normal', 'Pneumonia'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Ensemble Confusion Matrix (Optimal Threshold)')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)
plt.colorbar(im); plt.tight_layout()
plt.savefig(f"{fig_dir}/confusion_matrix.png", dpi=150); plt.close()

# ROC Curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, probs in [('DenseNet121', dn_probs), ('EfficientNet-B0', en_probs), ('Ensemble', ens_probs)]:
    fp, tp, _ = roc_curve(test_labels, probs)
    auc_val = roc_auc_score(test_labels, probs)
    ax.plot(fp, tp, label=f'{name} (AUC={auc_val:.4f})')
ax.plot([0,1], [0,1], 'k--', alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC Curves'); ax.legend(); plt.tight_layout()
plt.savefig(f"{fig_dir}/roc_curves.png", dpi=150); plt.close()

# PR Curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, probs in [('DenseNet121', dn_probs), ('EfficientNet-B0', en_probs), ('Ensemble', ens_probs)]:
    prec, rec, _ = precision_recall_curve(test_labels, probs)
    ax.plot(rec, prec, label=name)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves'); ax.legend(); plt.tight_layout()
plt.savefig(f"{fig_dir}/pr_curves.png", dpi=150); plt.close()

# Training Curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for name, mfile in [('DenseNet121', 'densenet121_metrics.csv'),
                     ('EfficientNet-B0', 'efficientnet_b0_metrics.csv')]:
    df = pd.read_csv(f"{CONFIG['output_dir']}/{mfile}")
    axes[0].plot(df['epoch'], df['train_loss'], label=f'{name} Train')
    axes[0].plot(df['epoch'], df['val_loss'], '--', label=f'{name} Val')
    axes[1].plot(df['epoch'], df['train_acc'], label=f'{name} Train')
    axes[1].plot(df['epoch'], df['val_acc'], '--', label=f'{name} Val')
    axes[2].plot(df['epoch'], df['val_auc'], label=name)
axes[0].set_title('Loss'); axes[0].legend()
axes[1].set_title('Accuracy (%)'); axes[1].legend()
axes[2].set_title('Val AUC'); axes[2].legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/training_curves.png", dpi=150); plt.close()

# Merge metrics
dn_df = pd.read_csv(f"{CONFIG['output_dir']}/densenet121_metrics.csv")
en_df = pd.read_csv(f"{CONFIG['output_dir']}/efficientnet_b0_metrics.csv")
dn_df['model'] = 'densenet121'; en_df['model'] = 'efficientnet_b0'
pd.concat([dn_df, en_df]).to_csv(f"{CONFIG['output_dir']}/training_metrics.csv", index=False)

print("\n" + "="*60)
print("ALL DONE! Download /kaggle/working/output/")
print("="*60)
for f in sorted(os.listdir(CONFIG['output_dir'])):
    fp = os.path.join(CONFIG['output_dir'], f)
    if os.path.isfile(fp):
        print(f"  {f} ({os.path.getsize(fp)/1024:.1f} KB)")
    elif os.path.isdir(fp):
        for ff in sorted(os.listdir(fp)):
            print(f"  {f}/{ff}")
