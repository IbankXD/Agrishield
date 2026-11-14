# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Primary users — Thai farmers and crop growers (เกษตรกรและผู้ปลูกพืชในไทย):** Inspecting leaves in the field or at home on mobile phones or desktop computers to quickly identify plant diseases across rice, tomato, corn, and potato crops and get actionable, eco-friendly biocontrol recipes without relying on toxic synthetic pesticides.
- **Secondary users — Academic evaluation committee & advisors (คณะกรรมการสอบโครงงานบูรณาการ CSI410, CSI411, CSI413, CSI414):** Evaluating diagnostic accuracy, explainability (Grad-CAM), data science methodology (stratified 70:30 split, class-weighted training, biological preservation of lesion color), and alignment with UN Sustainable Development Goal 12 (SDG 12).

## Product Purpose

AgriShield AI diagnoses plant leaf diseases from uploaded photos or live camera snapshots across 23 classes spanning 4 major economic crops (Rice, Tomato, Corn, Potato) and pairs every diagnosis with an immediate, practical Thai-language biological control (สารชีวภัณฑ์) and cultural management plan. Success means farmers can verify *why* the AI made its diagnosis via visual heatmap evidence and immediately apply safe biocontrol formulations (such as *Trichoderma harzianum*, *Bacillus subtilis*, or *Beauveria bassiana*) to reduce chemical pesticide residue under SDG 12.

## Positioning

Unlike black-box plant classification demos that stop at an English disease label or recommend hazardous chemical fungicides, AgriShield AI combines:
1. **Fine-tuned MobileNetV3 Large inference (99.49% Test Accuracy, 0.9919 Macro F1)** trained with cost-sensitive balanced class weights and biologically faithful geometric-only augmentations (preserving true lesion colors).
2. **Explainable AI (Grad-CAM Heatmap)** that visually highlights either necrotic/chlorotic lesion areas on diseased leaves or healthy chlorophyll blade regions on normal leaves, plus a **Top-2 Differential Diagnosis Advisory** whenever Top-1 confidence falls below 85%.
3. **Actionable Thai Biocontrol Advisor (SDG 12)** mapping all 23 classes to specific biological agents, mixing ratios, application timing, cultural field management, and environmental impact.

## Operating Context

- Used both on-site in agricultural plots via mobile/webcam capture (`HTML5 WebRTC`) and on desktop browsers via drag-and-drop image upload.
- Evaluated during the Week 13 Model Gate and Final Capstone Presentation for integrated courses **CSI410, CSI411, CSI413, and CSI414** (Academic Year 2569).
- Printable diagnostic report mode (`window.print()`) allows users or agricultural extension officers to save or hand out treatment sheets.

## Capabilities and Constraints

- **Supported Crops & Classes (23 classes total):**
  - **Tomato (มะเขือเทศ — 10 classes):** Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites (Two-spotted spider mite), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato mosaic virus, Healthy.
  - **Rice (ข้าว — 6 classes):** Bacterial Leaf Blight, Brown Spot, Leaf Blast, Leaf Scald, Sheath Blight, Healthy Rice Leaf.
  - **Corn / Maize (ข้าวโพด — 4 classes):** Cercospora leaf spot (Gray leaf spot), Common rust, Northern Leaf Blight, Healthy.
  - **Potato (มันฝรั่ง — 3 classes, including in-field images):** Early blight, Late blight, Healthy.
- **Technical Architecture:**
  - Model Engine: `web_app/model_engine.py` loading `best_agrishield_mobilenetv3.pth` (~17.1 MB) with PyTorch (`torch`, `torchvision`, `numpy`, `Pillow`) and studio white-background bias compensation.
  - Recommendation Database: `web_app/recommendation_db.py` (23-class Thai dictionary).
  - Web Server: Dual support for zero-dependency Python `http.server` (`web_app/server.py`, launched via `run_webapp.bat` on port 8000) and FastAPI (`web_app/app.py` with `/predict`, `/api/predict`, and `/api/classes`).
- **Scientific & Domain Constraints:**
  - Never alter leaf/lesion color profiles during preprocessing or display.
  - Always present a veterinary/agronomic disclaimer advising consultation with local agricultural extension officers (สำนักงานเกษตรอำเภอ) for severe outbreaks.

## Brand Commitments

- **Name:** AgriShield AI
- **Language & Voice:** Clear, respectful, accessible Thai (`ภาษาไทย`) paired with accurate scientific pathogen names (`Xanthomonas oryzae`, `Alternaria solani`, `Magnaporthe oryzae`, etc.) and international English class identifiers.
- **Project Attribution:** Group 32 (กลุ่มที่ 32) — Siwat Kaewnongsang (ศิวัตม์ แก้วหนองแสง) & Thanawat Narongwongwattana (ธนวรรธน์ ณรงค์วงศ์วัฒนา); Advisor: Asst. Prof. Dr. Worapong Rangseekasemkul (ผศ.ดร. วรพงศ์ รังสีเกษมกุล).

## Evidence on Hand

- Trained model checkpoint: `best_agrishield_mobilenetv3.pth` (verified 99.49% Test Accuracy, 0.9919 Macro F1 on 8,857 holdout test images from a 29,493-image dataset).
- Complete training & evaluation pipeline: `AgriShield_AI_Complete_Pipeline.ipynb`.
- Model Gate documentation & slides: `AgriShield_AI_Week13_Model_Gate_Summary.pdf` and `AgriShield_AI_Week13_Model_Gate.pptx`.
- Real test leaf uploads and generated Grad-CAM heatmaps in `web_app/static/uploads/`.
- Do not fabricate farmer testimonials, commercial certifications, or regulatory approvals not present in the repository.

## Product Principles

1. **Biological Truth Over Black-Box Labels:** Every diagnosis must show *where* the model looked (Grad-CAM) and warn the user with a Top-2 differential diagnosis when confidence is below 85%.
2. **Eco-First Actionability (SDG 12):** Treatment advice must prioritize biological control agents (BS, Trichoderma, Beauveria) with concrete mixing ratios and application times rather than generic chemical sprays.
3. **Field-to-Committee Clarity:** The interface must be effortless for a farmer uploading a leaf photo in 2 clicks while surfacing the rigorous scientific metrics and pathology details expected by academic reviewers.
4. **Zero-Friction Local Reliability:** The web application must run reliably out of the box via `run_webapp.bat` on local Windows machines without complex cloud dependencies.

## Accessibility & Inclusion

- High-contrast Thai typography with clear visual hierarchy suitable for outdoor/field viewing and projector presentations.
- Distinct visual encoding for healthy leaves (emerald/cyan confirmation) versus diseased leaves (warm amber/red lesion alerts) accompanied by explicit text labels so status is never conveyed by color alone.
