# ==============================================================================
# 🌾 AgriShield AI — Multi-Model Pipeline (4 Crop Specialist Models)
# สัดส่วนข้อมูลมาตรฐานสากล: Train 70% : Val 15% : Test 15% (Stratified Split)
# ==============================================================================
# 1. ข้าว (Rice Model)        : 6 คลาส
# 2. ข้าวโพด (Corn Model)     : 4 คลาส
# 3. มันฝรั่ง (Potato Model)   : 3 คลาส (รวมภาพแปลงจริงภายนอก muhammadardiputra)
# 4. มะเขือเทศ (Tomato Model)  : 10 คลาส
# ==============================================================================

# ------------------------------------------------------------------------------
# 🛠️ เซลล์ 0: ติดตั้งไลบรารีและเตรียมระบบ (Libraries & Setup)
# ------------------------------------------------------------------------------
!pip install kagglehub split-folders -q

import os
import time
import zipfile
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile
from collections import Counter
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import files

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ ติดตั้งและนำเข้าโมดูลทั้งหมดเรียบร้อยแล้ว! (ประมวลผลบน: {device})")


# ------------------------------------------------------------------------------
# 📦 เซลล์ 1: ดาวน์โหลดชุดข้อมูลทั้งหมดจาก Kaggle (เหมือนเดิม)
# ------------------------------------------------------------------------------
print("1. กำลังดาวน์โหลด PlantVillage Dataset...")
pv_path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")

color_dir = None
for root, dirs, files_in_dir in os.walk(pv_path):
    if os.path.basename(root).lower() == 'color':
        color_dir = root
        break
if not color_dir:
    color_dir = pv_path
print(f" • PlantVillage Color dir: {color_dir}")

print("\n2. กำลังดาวน์โหลด Rice Disease Dataset...")
rice_path = kagglehub.dataset_download("anshulm257/rice-disease-dataset")
rice_dir = rice_path
for root, dirs, files_in_dir in os.walk(rice_path):
    if any("blast" in d.lower() or "blight" in d.lower() for d in dirs):
        rice_dir = root
        break
print(f" • Rice dir: {rice_dir}")

print("\n3. กำลังดาวน์โหลด External Potato Dataset (muhammadardiputra)...")
ext_potato_path = kagglehub.dataset_download("muhammadardiputra/potato-leaf-disease-dataset")
print(f" • External Potato dir: {ext_potato_path}")

print("\n✅ ดาวน์โหลดครบทั้ง 3 แหล่งข้อมูลสำเร็จ!")


# ------------------------------------------------------------------------------
# 🌾 เซลล์ 2: รวมข้อมูลและทำความสะอาด (Clean & Resize 224x224)
# ------------------------------------------------------------------------------
clean_dir = "/content/plantvillage_clean_4crops"
target_size = (224, 224)
os.makedirs(clean_dir, exist_ok=True)

# 1. ดึง 3 พืชจาก PlantVillage (Tomato, Corn, Potato)
pv_keywords = ["tomato", "potato", "corn", "maize"]
print("🧹 1. กำลัง Clean & Resize ภาพจาก PlantVillage...")
for class_name in tqdm(os.listdir(color_dir), desc="PlantVillage"):
    class_path = os.path.join(color_dir, class_name)
    if os.path.isdir(class_path) and any(kw in class_name.lower() for kw in pv_keywords):
        dst_folder = os.path.join(clean_dir, class_name)
        os.makedirs(dst_folder, exist_ok=True)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_file = os.path.join(class_path, img_name)
                try:
                    with Image.open(src_file) as img:
                        img.verify()
                    with Image.open(src_file) as img:
                        img_rgb = img.convert("RGB")
                        img_resized = img_rgb.resize(target_size, Image.LANCZOS)
                        img_resized.save(os.path.join(dst_folder, img_name), quality=95)
                except Exception:
                    pass

# 2. ดึงข้าว (Rice Dataset)
print("🌾 2. กำลัง Clean & Resize ภาพข้าว...")
for class_name in tqdm(os.listdir(rice_dir), desc="Rice"):
    class_path = os.path.join(rice_dir, class_name)
    if os.path.isdir(class_path):
        target_class_name = f"Rice___{class_name.replace(' ', '_')}"
        dst_folder = os.path.join(clean_dir, target_class_name)
        os.makedirs(dst_folder, exist_ok=True)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_file = os.path.join(class_path, img_name)
                try:
                    with Image.open(src_file) as img:
                        img.verify()
                    with Image.open(src_file) as img:
                        img_rgb = img.convert("RGB")
                        img_resized = img_rgb.resize(target_size, Image.LANCZOS)
                        img_resized.save(os.path.join(dst_folder, img_name), quality=95)
                except Exception:
                    pass

# 3. เสริมภาพ Potato แปลงจริงจาก External Dataset
print("🥔 3. กำลังเสริมภาพมันฝรั่งแปลงจริง (External Potato)...")
potato_mapping = {
    'early_blight': 'Potato___Early_blight',
    'early blight': 'Potato___Early_blight',
    'late_blight': 'Potato___Late_blight',
    'late blight': 'Potato___Late_blight',
    'healthy': 'Potato___healthy'
}
for root, dirs, files_in_dir in os.walk(ext_potato_path):
    folder_name = os.path.basename(root).lower().strip()
    matched_target = None
    for key, target_folder in potato_mapping.items():
        if key in folder_name:
            matched_target = target_folder
            break
    if matched_target:
        dst_folder = os.path.join(clean_dir, matched_target)
        os.makedirs(dst_folder, exist_ok=True)
        for file in files_in_dir:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_file = os.path.join(root, file)
                try:
                    with Image.open(src_file) as img:
                        img.verify()
                    with Image.open(src_file) as img:
                        img_rgb = img.convert("RGB")
                        img_resized = img_rgb.resize(target_size, Image.LANCZOS)
                        img_resized.save(os.path.join(dst_folder, f"ext_{file}"), quality=95)
                except Exception:
                    pass

total_clean_imgs = sum([len(f) for r, d, f in os.walk(clean_dir)])
print(f"🎉 รวมภาพที่คลีนสมบูรณ์ทั้งหมด: {total_clean_imgs:,} ภาพ ใน {clean_dir}")


# ------------------------------------------------------------------------------
# ✂️ เซลล์ 3: แบ่งชุดข้อมูล Stratified Split 70% Train : 15% Val : 15% Test
# ------------------------------------------------------------------------------
import splitfolders
import shutil

split_output_dir = "/content/plantvillage_split"
if os.path.exists(split_output_dir):
    shutil.rmtree(split_output_dir)

print("✂️ กำลังแบ่งข้อมูลด้วย Stratified Split สัดส่วน 70% Train : 15% Val : 15% Test...")
splitfolders.ratio(clean_dir, output=split_output_dir, seed=42, ratio=(0.70, 0.15, 0.15), group_prefix=None, move=False)

print("\n" + "=" * 60)
print("📊 ผลการแบ่งชุดข้อมูลอย่างเป็นทางการ (70 : 15 : 15):")
print("=" * 60)
for split in ['train', 'val', 'test']:
    p = os.path.join(split_output_dir, split)
    if os.path.exists(p):
        count = sum([len(files) for r, d, files in os.walk(p)])
        pct = (count / total_clean_imgs) * 100.0
        print(f" • {split.upper():<5} : {count:>6,} ภาพ ({pct:.1f}%)")
print("=" * 60)


# ------------------------------------------------------------------------------
# 🧠 เซลล์ 4: กำหนด CropSubDataset & ฟังก์ชันเทรนโมเดลเฉพาะพืช (Reusable)
# ------------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Custom Dataset คัดเฉพาะคลาสของพืชแต่ละชนิดโดยตรงจากโฟลเดอร์ ไม่ต้องก๊อปปี้ไฟล์ซ้ำ
class CropSubDataset(Dataset):
    def __init__(self, base_folder, crop_prefix, transform=None):
        self.transform = transform
        full_ds = datasets.ImageFolder(base_folder)
        self.classes = sorted([c for c in full_ds.classes if c.startswith(crop_prefix)])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.samples = []
        for path, old_idx in full_ds.samples:
            cname = full_ds.classes[old_idx]
            if cname in self.class_to_idx:
                self.samples.append((path, self.class_to_idx[cname]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

# กำหนดสเปกของ 4 พืชหลัก
CROP_SPECS = {
    'Rice': {
        'prefix': 'Rice___',
        'checkpoint': 'best_agrishield_rice.pth',
        'display_name': '🌾 ข้าว (Rice)'
    },
    'Corn': {
        'prefix': 'Corn_(maize)___',
        'checkpoint': 'best_agrishield_corn.pth',
        'display_name': '🌽 ข้าวโพด (Corn)'
    },
    'Potato': {
        'prefix': 'Potato___',
        'checkpoint': 'best_agrishield_potato.pth',
        'display_name': '🥔 มันฝรั่ง (Potato)'
    },
    'Tomato': {
        'prefix': 'Tomato___',
        'checkpoint': 'best_agrishield_tomato.pth',
        'display_name': '🍅 มะเขือเทศ (Tomato)'
    }
}

print("✅ ออกแบบ CropSubDataset รองรับ Train / Val / Test (70:15:15) เรียบร้อย!")


# ------------------------------------------------------------------------------
# 🔥 เซลล์ 5: ฟังก์ชันฝึกสอนโมเดลสำหรับ 1 พืช (70% Train, 15% Val, 15% Test)
# ------------------------------------------------------------------------------
def train_single_crop_model(crop_key, epochs=10, batch_size=32):
    spec = CROP_SPECS[crop_key]
    prefix = spec['prefix']
    ckpt_name = spec['checkpoint']
    display_name = spec['display_name']

    print("\n" + "=" * 75)
    print(f"🚀 เริ่มต้นการเทรนโมเดลเฉพาะพืช: {display_name}")
    print("=" * 75)

    # 1. สร้าง DataLoaders ทั้ง 3 ชุด (Train 70% : Val 15% : Test 15%)
    train_ds = CropSubDataset(os.path.join(split_output_dir, 'train'), prefix, transform=train_transforms)
    val_ds   = CropSubDataset(os.path.join(split_output_dir, 'val'),   prefix, transform=eval_transforms)
    test_ds  = CropSubDataset(os.path.join(split_output_dir, 'test'),  prefix, transform=eval_transforms)
    
    num_classes = len(train_ds.classes)
    print(f" • จำนวนคลาสของ {crop_key}: {num_classes} คลาส")
    print(f" • สัดส่วนภาพ 70:15:15: Train = {len(train_ds):,} | Val = {len(val_ds):,} | Test = {len(test_ds):,}")
    for i, c in enumerate(train_ds.classes):
        print(f"   [{i}] {c.replace(prefix, '')}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. ปรับแต่ง MobileNetV3 Large
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # 3. Loss & Optimizer (ไม่ต้องใช้ Class Weights รุนแรง เพราะแยกพืชแล้ว)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. ลูปการฝึกสอน (วัดผลและเลือก Best Checkpoint บนชุด Validation 15%)
    best_val_acc = 0.0
    start_t = time.time()

    for ep in range(epochs):
        model.train()
        r_loss, correct_tr, total_tr = 0.0, 0, 0
        for imgs, lbls in tqdm(train_loader, desc=f"{crop_key} Epoch {ep+1}/{epochs} [Train]", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, lbls)
            loss.backward()
            optimizer.step()

            r_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outs, 1)
            correct_tr += (preds == lbls).sum().item()
            total_tr += lbls.size(0)

        tr_loss = r_loss / total_tr
        tr_acc  = correct_tr / total_tr

        # Validation บนชุด Val 15% (เพื่อเลือกโมเดลที่ดีที่สุดอย่างเป็นกลาง)
        model.eval()
        v_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outs = model(imgs)
                loss = criterion(outs, lbls)
                v_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outs, 1)
                correct_val += (preds == lbls).sum().item()
                total_val += lbls.size(0)

        val_loss = v_loss / total_val
        val_acc  = correct_val / total_val
        scheduler.step()

        print(f"  Epoch {ep+1:02d}/{epochs:02d} | Train Acc: {tr_acc*100:.2f}% | Val Loss: {val_loss:.4f} Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_name)

    dur = time.time() - start_t
    print(f"⭐ จบการเทรน {display_name} ใน {dur/60:.2f} นาที | Best Val Accuracy: {best_val_acc*100:.2f}%")

    # 5. ประเมินผลลัพธ์รอบสุดท้ายบน "Unseen Test Set 15%" (ภาพที่ไม่เคยผ่านตาทั้งตอน Train และ Val)
    print(f"📊 กำลังวัดผลการทดสอบรอบสุดท้ายบน Unseen Test Set (15%)...")
    model.load_state_dict(torch.load(ckpt_name))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outs = model(imgs)
            _, preds = torch.max(outs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())

    rep = classification_report(all_labels, all_preds, target_names=[c.replace(prefix, '') for c in train_ds.classes], output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'crop_key': crop_key,
        'display_name': display_name,
        'classes': train_ds.classes,
        'val_accuracy': best_val_acc * 100.0,
        'test_accuracy': rep['accuracy'] * 100.0,
        'macro_f1': rep['macro avg']['f1-score'],
        'weighted_f1': rep['weighted avg']['f1-score'],
        'report': rep,
        'cm': cm,
        'checkpoint': ckpt_name
    }


# ------------------------------------------------------------------------------
# 🚀 เซลล์ 6: รันการเทรนทั้ง 4 โมเดล (Rice, Corn, Potato, Tomato)
# ------------------------------------------------------------------------------
results_all = {}

for crop in ['Rice', 'Corn', 'Potato', 'Tomato']:
    results_all[crop] = train_single_crop_model(crop, epochs=10, batch_size=32)

print("\n🎉 ฝึกสอนและประเมินผลครบทั้ง 4 โมเดลเรียบร้อยแล้ว!")


# ------------------------------------------------------------------------------
# 📊 เซลล์ 7: ตารางสรุป Benchmark 4 พืช & กราฟ Confusion Matrix (สำหรับสไลด์)
# ------------------------------------------------------------------------------
print("=" * 90)
print("📊 [แคปใส่สไลด์: ตารางเปรียบเทียบผลลัพธ์ 4 โมเดล สัดส่วน 70% Train : 15% Val : 15% Test]")
print("=" * 90)

summary_rows = []
for crop, res in results_all.items():
    summary_rows.append({
        'ชนิดพืช (Crop)': res['display_name'],
        'จำนวนคลาส (Classes)': len(res['classes']),
        'Val Accuracy (%)': f"{res['val_accuracy']:.2f}%",
        'Test Accuracy (%)': f"{res['test_accuracy']:.2f}%",
        'Test Macro F1': f"{res['macro_f1']:.4f}",
        'Weighted F1': f"{res['weighted_f1']:.4f}",
        'Model Size': '~16.4 MB',
        'Checkpoint File': res['checkpoint']
    })

df_summary = pd.DataFrame(summary_rows)
display(df_summary)

# วาด Confusion Matrix รวม 4 พืชในภาพเดียว (2x2 Grid)
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
crop_list = ['Rice', 'Corn', 'Potato', 'Tomato']

for idx, crop in enumerate(crop_list):
    ax = axes[idx // 2, idx % 2]
    res = results_all[crop]
    clean_labels = [c.replace(CROP_SPECS[crop]['prefix'], '').replace('_', ' ') for c in res['classes']]
    
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=clean_labels, yticklabels=clean_labels)
    ax.set_title(f"{res['display_name']} — Test Acc: {res['test_accuracy']:.2f}% (Macro F1: {res['macro_f1']:.4f})", fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label (AI ทำนาย)', fontsize=10)
    ax.set_ylabel('True Label (คำตอบจริง)', fontsize=10)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)

plt.tight_layout()
plt.savefig("agrishield_4_models_confusion_matrices_70_15_15.png", dpi=300, bbox_inches='tight')
plt.show()
print("🖼️ บันทึกรูป agrishield_4_models_confusion_matrices_70_15_15.png สำหรับสไลด์เรียบร้อย!")


# ------------------------------------------------------------------------------
# 💾 เซลล์ 8: รวมไฟล์ Checkpoints ทั้ง 4 โมเดลและดาวน์โหลดลงเครื่อง
# ------------------------------------------------------------------------------
zip_filename = "agrishield_4_crop_models_70_15_15.zip"
print(f"📦 กำลังรวมไฟล์ Checkpoint ทั้ง 4 พืชลงใน {zip_filename}...")

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for crop, res in results_all.items():
        ckpt = res['checkpoint']
        if os.path.exists(ckpt):
            zipf.write(ckpt)
            print(f" • รวมไฟล์: {ckpt}")
    if os.path.exists("agrishield_4_models_confusion_matrices_70_15_15.png"):
        zipf.write("agrishield_4_models_confusion_matrices_70_15_15.png")

print(f"\n⬇️ กำลังเริ่มดาวน์โหลด {zip_filename} ลงเครื่องของคุณ...")
files.download(zip_filename)
