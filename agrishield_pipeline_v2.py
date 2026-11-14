# ==============================================================================
# 🌾 AgriShield AI — Data Pipeline & Cross-Validation Architecture (v2)
# CSI411 / CSI410 | กลุ่ม 32
# ------------------------------------------------------------------------------
# สถาปัตยกรรมข้อมูลตามฟีดแบ็กอาจารย์และ อ.ตี๋:
# 1. รวบรวม 4 พืชเศรษฐกิจ: Tomato, Corn, Potato, Rice (23 คลาส)
# 2. Data Cleaning: ตรวจสอบความสมบูรณ์ไฟล์ (verify) และ Resize 224x224
# 3. Stratified Splitting: 70% Train / 30% Test (Hold-out Unseen Set)
# 4. Model Selection & Tuning: 5-Fold Stratified Cross-Validation บนชุด Train 70%
# 5. Safe Geometric Augmentation: ไม่ดัดแปลงสี/ไม่แปลงขาวดำ เพื่อรักษารอยโรค
# ==============================================================================

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1: ติดตั้งไลบรารีและเตรียมระบบ                                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝
!pip install kagglehub split-folders -q

import os
import shutil
import hashlib
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from collections import defaultdict
from tqdm.notebook import tqdm

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

ImageFile.LOAD_TRUNCATED_IMAGES = True
print("✅ เตรียม Environment และโมดูลเรียบร้อยแล้ว!")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2: ดาวน์โหลดและเชื่อมโยงชุดข้อมูล 4 พืช                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝
print("📦 กำลังดาวน์โหลดชุดข้อมูล PlantVillage (Tomato, Corn, Potato)...")
pv_path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")

# ค้นหาโฟลเดอร์ color ของ PlantVillage
pv_color_dir = None
for root, dirs, files in os.walk(pv_path):
    if os.path.basename(root).lower() == 'color':
        pv_color_dir = root
        break
if not pv_color_dir:
    pv_color_dir = pv_path
print(f"  • PlantVillage Color dir: {pv_color_dir}")

print("📦 กำลังดาวน์โหลดชุดข้อมูล Rice Disease Dataset (Rice)...")
rice_path = kagglehub.dataset_download("anshulm257/rice-disease-dataset")

# ค้นหาโฟลเดอร์คลาสของข้าว
rice_dir = rice_path
for root, dirs, files in os.walk(rice_path):
    if any("blast" in d.lower() or "blight" in d.lower() for d in dirs):
        rice_dir = root
        break
print(f"  • Rice dir: {rice_dir}")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3: รวบรวมข้อมูลและทำ Data Cleaning (Integrity & Resize 224x224)       ║
# ╚════════════════════════════════════════════════════════════════════════════╝
clean_dataset_dir = "/content/agrishield_clean_dataset"
target_size = (224, 224)
raw_records = []

# --- 1. สแกน PlantVillage (คัดเฉพาะ Tomato, Corn, Potato) ---
pv_keywords = ["tomato", "potato", "corn", "maize"]
for cls_name in sorted(os.listdir(pv_color_dir)):
    src_cls_path = os.path.join(pv_color_dir, cls_name)
    if os.path.isdir(src_cls_path) and any(kw in cls_name.lower() for kw in pv_keywords):
        imgs = [f for f in os.listdir(src_cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(imgs) > 0:
            parts = cls_name.split('___')
            crop = parts[0].replace('_', ' ')
            condition = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
            status = 'Healthy' if 'healthy' in cls_name.lower() else 'Diseased'
            raw_records.append({
                'Class_Full': cls_name,
                'Crop': crop,
                'Condition': condition,
                'Status': status,
                'Source_Dir': src_cls_path
            })

# --- 2. สแกน Rice Dataset ---
for cls_name in sorted(os.listdir(rice_dir)):
    src_cls_path = os.path.join(rice_dir, cls_name)
    if os.path.isdir(src_cls_path):
        imgs = [f for f in os.listdir(src_cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(imgs) > 0:
            formatted_cls = f"Rice___{cls_name.replace(' ', '_')}"
            status = 'Healthy' if 'healthy' in cls_name.lower() else 'Diseased'
            raw_records.append({
                'Class_Full': formatted_cls,
                'Crop': 'Rice',
                'Condition': cls_name.replace('_', ' '),
                'Status': status,
                'Source_Dir': src_cls_path
            })

df_classes = pd.DataFrame(raw_records)
print(f"📊 พบรายการทั้งหมด: {len(df_classes)} คลาส จาก 4 ชนิดพืช")

# --- 3. ดำเนินการ Data Cleaning และ Resize ---
print("\n🧹 กำลังดำเนินการ Data Cleaning & Resizing (224x224)...")
clean_stats = []
corrupted_files = []

for idx, row in tqdm(df_classes.iterrows(), total=len(df_classes), desc="Cleaning"):
    dst_cls_dir = os.path.join(clean_dataset_dir, row['Class_Full'])
    os.makedirs(dst_cls_dir, exist_ok=True)
    
    valid_count = 0
    for img_file in os.listdir(row['Source_Dir']):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        src_img_path = os.path.join(row['Source_Dir'], img_file)
        try:
            # ตรวจสอบไฟล์เสีย
            with Image.open(src_img_path) as img:
                img.verify()
            # แปลงเป็น RGB และ Resize
            with Image.open(src_img_path) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize(target_size, Image.LANCZOS)
                dst_img_path = os.path.join(dst_cls_dir, img_file)
                img_resized.save(dst_img_path, quality=95)
                valid_count += 1
        except Exception as e:
            corrupted_files.append((src_img_path, str(e)))
            
    clean_stats.append(valid_count)

df_classes['Clean_Image_Count'] = clean_stats
total_cleaned = sum(clean_stats)

print("=" * 60)
print(f"✅ Data Cleaning เสร็จสิ้นสมบูรณ์:")
print(f"  • ภาพที่ผ่านเกณฑ์ทั้งหมด : {total_cleaned:,} ภาพ")
print(f"  • ไฟล์เสียหายที่ถูกตัดทิ้ง : {len(corrupted_files)} ไฟล์")
print(f"  • โฟลเดอร์ปลายทาง       : {clean_dataset_dir}")
print("=" * 60)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4: Stratified Splitting (70% Train : 30% Test)                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝
import splitfolders

output_split_dir = "/content/agrishield_split_70_30"
if os.path.exists(output_split_dir):
    shutil.rmtree(output_split_dir)

print("✂️ ดำเนินการ Stratified Splitting อัตราส่วน 70 : 30 (Train : Test)...")
# อัตราส่วน 70% Train, 0% Val, 30% Test (Hold-out)
splitfolders.ratio(
    clean_dataset_dir,
    output=output_split_dir,
    seed=42,
    ratio=(0.70, 0.0, 0.30),
    group_prefix=None,
    move=False
)

train_dir = os.path.join(output_split_dir, "train")
test_dir = os.path.join(output_split_dir, "val")  # splitfolders เก็บชุดหลังไว้ที่ชื่อ val

# เปลี่ยนชื่อโฟลเดอร์ให้สื่อความหมายชัดเจนว่าเป็น test
final_test_dir = os.path.join(output_split_dir, "test")
if os.path.exists(test_dir):
    os.rename(test_dir, final_test_dir)

train_count = sum([len(files) for r, d, files in os.walk(train_dir)])
test_count = sum([len(files) for r, d, files in os.walk(final_test_dir)])

print("\n📊 สรุปผลการแบ่งชุดข้อมูล (Dataset Splitting 70:30):")
print(f"  • 🟢 Training Set (70%)  : {train_count:>6,} ภาพ (สำหรับ 5-Fold Cross-Validation)")
print(f"  • 🔴 Test Set (30%)      : {test_count:>6,} ภาพ (สำหรับ Unseen Final Evaluation)")
print(f"  • ⚪ Total Images        : {train_count + test_count:>6,} ภาพ")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5: Data Augmentation ที่ปลอดภัยต่อสีรอยโรค (Safe Geometric Only)     ║
# ╚════════════════════════════════════════════════════════════════════════════╝
# ⚠️ ตามคำแนะนำของอาจารย์และ อ.ตี๋:
# - ห้ามใช้ RandomGrayscale หรือ ColorJitter ที่สุ่มค่ารุนแรง
# - ใช้เฉพาะ Geometric Transform เพื่อรักษารงควัตถุและลักษณะสีของแผลโรคพืช

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Transform สำหรับ Train Fold (มี Geometric Augmentation เล็กน้อยเพื่อความคงทน)
safe_train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Transform สำหรับ Validation Fold และ Test Set (ไม่มี Augmentation โดยเด็ดขาด)
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

print("✅ กำหนดนิยาม Transform ปลอดภัยสำเร็จ: รักษาสีของรอยโรค ไม่ทำลาย Diagnostic Features")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6: สถาปัตยกรรม 5-Fold Stratified Cross-Validation (บน Train 70%)     ║
# ╚════════════════════════════════════════════════════════════════════════════╝
# โหลดข้อมูล Train Set ทั้งหมด
train_full_dataset = datasets.ImageFolder(train_dir)
class_names = train_full_dataset.classes
targets = [s[1] for s in train_full_dataset.samples]

# สร้างตัวสร้าง Fold แบบ Stratified
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"🚀 เตรียมโครงสร้าง 5-Fold Stratified Cross-Validation บนชุด Train ({len(train_full_dataset):,} ภาพ):")
print("-" * 65)

fold_loaders = []
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
    # แยก Subset ตามดัชนีของแต่ละ Fold
    # ชุด Train ในแต่ละ Fold จะใช้ safe_train_transforms
    # ชุด Validation ในแต่ละ Fold จะใช้ eval_transforms
    train_subset = Subset(datasets.ImageFolder(train_dir, transform=safe_train_transforms), train_idx)
    val_subset   = Subset(datasets.ImageFolder(train_dir, transform=eval_transforms), val_idx)
    
    loader_train = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    loader_val   = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    fold_loaders.append((loader_train, loader_val))
    print(f"  • Fold {fold+1}: Train = {len(train_subset):>5,} ภาพ ({len(train_subset)/len(train_full_dataset)*100:.1f}%) | "
          f"Validation = {len(val_subset):>5,} ภาพ ({len(val_subset)/len(train_full_dataset)*100:.1f}%)")

# เตรียม Test Loader สำหรับ Final Test (Unseen 30%)
test_dataset = datasets.ImageFolder(final_test_dir, transform=eval_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

print("-" * 65)
print(f"🎯 Final Hold-out Test Loader: {len(test_dataset):,} ภาพ (ทดสอบรอบสุดท้าย)")
print(f"🏆 Data Pipeline สมบูรณ์แบบ 100% พร้อมเชื่อมต่อเข้าสู่การเทรน Model ต่อไปครับ!")
