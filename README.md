# AgriShield AI V2 🛡️🌿
> ระบบผู้เชี่ยวชาญ AI วินิจฉัยโรคพืช 4 ชนิดเศรษฐกิจ (ข้าว, มะเขือเทศ, ข้าวโพด, มันฝรั่ง) พร้อม Explainable AI (Grad-CAM) และคำแนะนำการใช้สารชีวภัณฑ์ (SDG 12)

---

## 📌 ภาพรวมสถาปัตยกรรม (Architecture Overview)

AgriShield AI V2 ปรับปรุงจากระบบเดิมเป็น **สถาปัตยกรรมโมเดลผู้เชี่ยวชาญเฉพาะทาง (4 Modular Specialist Models)** โดยแบ่งโมเดลตามชนิดพืชเพื่อความแม่นยำสูงสุด ลดการสับสนข้ามชนิดพืช:

1. **🌾 ข้าว (Rice Specialist):** 6 คลาส (Test Acc: 98.92%)
2. **🌽 ข้าวโพด (Corn Specialist):** 4 คลาส (Test Acc: 99.10%)
3. **🥔 มันฝรั่ง (Potato Specialist):** 3 คลาส (Test Acc: 100.00%)
4. **🍅 มะเขือเทศ (Tomato Specialist):** 10 คลาส (Test Acc: 99.87%)

* **Backbone:** MobileNetV3 Large (Pretrained บน ImageNet)
* **Dataset Partition:** Stratified Split 70% Train : 15% Validation : 15% Test
* **Class Weights:** คำนวณ Inverse Class Frequency เพื่อลดปัญหา Class Imbalance
* **Explainable AI:** Grad-CAM (Target Layer: `features[-1]`) ชี้เป้ารอยโรคเปรียบเทียบกับภาพต้นฉบับ
* **Biocontrol Knowledge Base:** ฐานข้อมูลสารชีวภัณฑ์และวิธีป้องกันกำจัดโรคภาษาไทย 23 คลาส อ้างอิงกรมวิชาการเกษตร

---

## 🚀 วิธีการติดตั้งและรันระบบ (Quick Start)

### 1. Requirements
- Python 3.10+ (รองรับ Python 3.14)
- PyTorch & Torchvision
- Pillow, NumPy

### 2. รันแอปพลิเคชัน
คลิกดับเบิลคลิกที่ไฟล์:
```cmd
run_v2.bat
```
หรือรันผ่านคำสั่ง Terminal:
```bash
cd V2
python server.py 8001
```
เปิดเบราว์เซอร์ที่: **http://127.0.0.1:8001**

---

## 📂 โครงสร้างไดเรกทอรี (Directory Structure)

```
Agrishield/
├── run_v2.bat              # Script รันระบบ V2 จาก Root
├── README.md               # เอกสารประกอบโครงการ
├── PRODUCT.md              # ข้อมูลสเปกและผลิตภัณฑ์
└── V2/                     # ระบบแอปพลิเคชัน AgriShield AI V2
    ├── server.py           # Web Server (Python http.server)
    ├── model_engine_v2.py  # Model Engine โหลด 4 โมเดล + Grad-CAM
    ├── recommendation_db.py# ฐานข้อมูลสารชีวภัณฑ์ 23 คลาส
    ├── run_v2.bat          # สคริปต์เปิดรัน V2 บน Port 8001
    ├── static/             # Assets, CSS, Confusion Matrices, Samples
    └── templates/          # Modern Dark Emerald UI
        ├── index.html      # หน้าจออัปโหลดและเลือกชนิดพืช
        └── result.html     # หน้าจอแสดงผลวินิจฉัยและ Grad-CAM
```

---

## 👥 ผู้พัฒนา (Development Team)
- **กลุ่มที่ 32 (Group 32)** — โครงงานบูรณาการหลักสูตรวิทยาการคอมพิวเตอร์ (CSI410, CSI411, CSI413, CSI414)
- **อาจารย์ที่ปรึกษา:** ผศ.ดร. วรพงศ์ รังสีเกษมกุล
