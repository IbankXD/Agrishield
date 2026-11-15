# AgriShield AI V2 🛡️🌿
> ระบบผู้เชี่ยวชาญ AI วินิจฉัยโรคพืช 4 ชนิดเศรษฐกิจ (ข้าว, มะเขือเทศ, ข้าวโพด, มันฝรั่ง) พร้อม Explainable AI (Grad-CAM) และคำแนะนำการใช้สารชีวภัณฑ์ (SDG 12)

---

## 📌 ภาพรวมสถาปัตยกรรม (Architecture Overview)

AgriShield AI V2 ทำงานด้วย **สถาปัตยกรรมโมเดลผู้เชี่ยวชาญเฉพาะทาง (4 Modular Specialist Models)** แบ่งโมเดลตามชนิดพืชเพื่อความแม่นยำสูงสุด ลดการสับสนข้ามชนิดพืช:

1. **🌾 ข้าว (Rice Specialist):** 6 คลาสโรค (ขอบใบแห้ง, ใบจุดสีน้ำตาล, โรคไหม้, ใบขีดไหม้, กาบใบแห้ง, ใบปกติ)
2. **🌽 ข้าวโพด (Corn Specialist):** 4 คลาสโรค (ราสนิม, ใบจุดสีเทา, ใบไหม้แผลใหญ่, ใบปกติ)
3. **🥔 มันฝรั่ง (Potato Specialist):** 3 คลาสโรค (ใบไหม้เช้า Early Blight, ใบไหม้สาย Late Blight, ใบปกติ)
4. **🍅 มะเขือเทศ (Tomato Specialist):** 10 คลาสโรค (ใบจุดแบคทีเรีย, ไวรัสใบหงิกเหลือง TYLCV, ใบจุดเซพโทเรีย ฯลฯ)

* **Backbone:** MobileNetV3 Large (Pretrained บน ImageNet)
* **Dataset Partition:** Stratified Split 70% Train : 15% Validation : 15% Test
* **Explainable AI:** Grad-CAM (Target Layer: `features[-1]`) วิเคราะห์พื้นที่ความสนใจของโมเดล
* **Biocontrol Knowledge Base:** ฐานข้อมูลสารชีวภัณฑ์และวิธีป้องกันกำจัดโรคภาษาไทย 23 คลาส อ้างอิงกรมวิชาการเกษตร (SDG 12)
* **Dual Theme UI:** สลับโหมด ☀️ **ธีมเขียว-ขาว (Emerald White)** และ 🌙 **ธีมเขียว-ดำ (Dark Emerald)** ได้ในคลิกเดียว

---

## 🚀 ขั้นตอนการดึงโค้ดไปติดตั้งและรันในเครื่อง (Setup & How to Run)

สำหรับเพื่อนร่วมทีมหรือผู้ที่ต้องการนำโปรเจกต์นี้ไปรันในเครื่องของตนเอง ทำตามขั้นตอนด้านล่างนี้ได้เลย:

### 1. ดึงโค้ดโปรเจกต์ลงเครื่อง (Clone / Pull)

* **กรณีดึงโปรเจกต์ครั้งแรก (First-Time Setup):**
  เปิด Terminal หรือ Git Bash แล้วพิมพ์คำสั่ง:
  ```bash
  git clone https://github.com/IbankXD/Agrishield.git
  cd Agrishield
  ```

* **กรณีมีโปรเจกต์ในเครื่องอยู่แล้ว ให้ดึงเวอร์ชันล่าสุดที่มีไฟล์โมเดลครบ:**
  ```bash
  git pull origin main
  ```

---

### 2. ติดตั้งแพ็กเกจที่จำเป็น (Install Dependencies)

เครื่องที่จะรันต้องติดตั้ง **Python 3.10 ขึ้นไป** (รองรับถึง Python 3.14)  
จากนั้นเปิด Command Prompt (cmd) หรือ Terminal ในโฟลเดอร์โปรเจกต์ แล้วติดตั้งไลบรารี:
```bash
pip install -r V2/requirements.txt
```
*(ไฟล์นี้จะติดตั้ง `torch`, `torchvision`, `numpy`, `pillow` อัตโนมัติ เพื่อให้โมเดล AI โหลดได้จริงและไม่ตกไปใช้ Fallback Mode)*

---

### 3. รันแอปพลิเคชัน (Start Application)

สามารถเลือกวิธีรันได้ 2 วิธี:

* **วิธีที่ 1 (แนะนำ - สะดวกที่สุด):**
  ดับเบิลคลิกที่ไฟล์:
  ```cmd
  run_v2.bat
  ```
  *(ระบบจะตรวจหา Python ในเครื่องและเปิดเบราว์เซอร์ให้อัตโนมัติ)*

* **วิธีที่ 2 (ผ่าน Command Line / Terminal):**
  ```bash
  cd V2
  python server.py 8001
  ```

เปิดเบราว์เซอร์ไปที่: **http://127.0.0.1:8001**

---

### 4. ตรวจสอบความถูกต้องเมื่อระบบเริ่มทำงาน

เมื่อรันสำเร็จ หน้าจอคอนโซล Terminal จะต้องแสดงข้อความว่าโหลดโมเดลครบทั้ง 5 ตัว ดังนี้:
```text
✅ V2 Engine: โหลด RICE Specialist (6 คลาส) สำเร็จ
✅ V2 Engine: โหลด CORN Specialist (4 คลาส) สำเร็จ
✅ V2 Engine: โหลด POTATO Specialist (3 คลาส) สำเร็จ
✅ V2 Engine: โหลด TOMATO Specialist (10 คลาส) สำเร็จ
✅ V2 Engine: โหลด Unified 23-Class Router สำเร็จ
```
> **หมายเหตุ:** หากไม่พบข้อความข้างต้น หรือขึ้นว่า `Fallback Engine` กรุณาตรวจสอบว่ามีไฟล์ `.pth` ครบทั้ง 5 ตัวอยู่ในโฟลเดอร์ `V2/models/` และรันคำสั่ง `pip install -r V2/requirements.txt` เรียบร้อยแล้ว

---

## 📂 โครงสร้างไดเรกทอรี (Directory Structure)

```
Agrishield/
├── run_v2.bat              # Script รันระบบ V2 จาก Root (ค้นหา Python อัตโนมัติ)
├── README.md               # เอกสารคู่มือการติดตั้งและรันระบบ
├── PRODUCT.md              # สเปกผลิตภัณฑ์และข้อมูลโมเดล
└── V2/                     # ซอร์สโค้ดระบบ AgriShield AI V2
    ├── server.py           # Web Server (Python http.server บน Port 8001)
    ├── model_engine_v2.py  # AI Model Engine (Multi-Model + Grad-CAM + OOD Filter)
    ├── recommendation_db.py# ฐานข้อมูลสารชีวภัณฑ์ 23 คลาส (SDG 12)
    ├── requirements.txt    # รายการแพ็กเกจ Python (torch, torchvision, numpy, pillow)
    ├── run_v2.bat          # สคริปต์คลิกรันเฉพาะโฟลเดอร์ V2
    ├── models/             # โฟลเดอร์เก็บน้ำหนักโมเดล PyTorch (.pth) ครบทั้ง 5 ตัว
    │   ├── best_agrishield_rice.pth
    │   ├── best_agrishield_corn.pth
    │   ├── best_agrishield_potato.pth
    │   ├── best_agrishield_tomato.pth
    │   └── best_agrishield_mobilenetv3.pth
    ├── static/             # ไฟล์รูปภาพตัวอย่าง, แผนผังผลลัพธ์ และ assets
    └── templates/          # หน้าจอ Modern Split-Screen & Theme Switcher
        ├── index.html      # หน้าจอเลือกพืช, ถ่ายภาพ/อัปโหลด และสลับธีม
        └── result.html     # หน้าจอแสดงผลวินิจฉัย, Grad-CAM และสารชีวภัณฑ์
```

---

## 🌿 มาตรฐานความปลอดภัยชีวภาพ (SDG 12)
โครงการนี้พัฒนาขึ้นเพื่อสนับสนุน **เป้าหมายการพัฒนาที่ยั่งยืนที่ 12 (Responsible Consumption and Production)** โดยทุกผลการวินิจฉัยจะแนะนำสารชีวภัณฑ์ปลอดภัย (เช่น เชื้อราไตรโคเดอร์มา *Trichoderma harzianum*, แบคทีเรียบาซิลลัส *Bacillus subtilis*) เพื่อลดการพึ่งพาสารเคมีสังเคราะห์ทางการเกษตร
