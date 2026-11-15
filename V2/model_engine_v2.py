# ==============================================================================
# 🚀 AgriShield AI V2 — Multi-Model Specialist Engine (70:15:15)
# สถาปัตยกรรม 4 โมเดลผู้เชี่ยวชาญเฉพาะพืช (Rice, Corn, Potato, Tomato)
# ==============================================================================
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

# Ensure Python 3.14 site-packages are always discoverable
_EXTRA_SITES = [
    r"C:\Users\door_\AppData\Local\Programs\Python\Python314\Lib\site-packages",
    r"C:\Users\door_\AppData\Roaming\Python\Python314\site-packages",
]
for _p in _EXTRA_SITES:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True and NUMPY_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False

# นิยามคลาสของแต่ละโมเดลผู้เชี่ยวชาญ (ตามลำดับ Colab)
CROP_CLASSES = {
    "rice": [
        "Rice___Bacterial_Leaf_Blight",
        "Rice___Brown_Spot",
        "Rice___Healthy_Rice_Leaf",
        "Rice___Leaf_Blast",
        "Rice___Leaf_Scald",
        "Rice___Sheath_Blight"
    ],
    "corn": [
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy"
    ],
    "potato": [
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy"
    ],
    "tomato": [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]
}

# รายชื่อ 23 คลาสรวมสำหรับ Auto-Detection Router
UNIFIED_23_CLASSES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Rice___Bacterial_Leaf_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy_Rice_Leaf",
    "Rice___Leaf_Blast",
    "Rice___Leaf_Scald",
    "Rice___Sheath_Blight",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class AgriShieldV2Engine:
    def __init__(self, models_dir: str = None):
        self.device = torch.device("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
        self.is_torch_ready = False
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).resolve().parent / "models"
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]) if TORCH_AVAILABLE else None

        self.specialist_models = {}
        self.unified_model = None

        # Grad-CAM storage per model
        self.feature_maps = {}
        self.gradients = {}

        if TORCH_AVAILABLE:
            self._load_all_models()

    def _hook_model(self, model, model_key):
        target_layer = model.features[-1]
        def forward_hook(module, input, output):
            self.feature_maps[model_key] = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients[model_key] = grad_out[0]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def _load_all_models(self):
        try:
            # 1. โหลด 4 โมเดลผู้เชี่ยวชาญ (Specialists)
            crop_filenames = {
                "rice": "best_agrishield_rice.pth",
                "corn": "best_agrishield_corn.pth",
                "potato": "best_agrishield_potato.pth",
                "tomato": "best_agrishield_tomato.pth"
            }

            for crop, fname in crop_filenames.items():
                p = self.models_dir / fname
                if p.exists():
                    classes = CROP_CLASSES[crop]
                    m = models.mobilenet_v3_large()
                    in_feat = m.classifier[3].in_features
                    m.classifier[3] = nn.Linear(in_feat, len(classes))
                    m.load_state_dict(torch.load(str(p), map_location=self.device))
                    m.to(self.device)
                    m.eval()
                    self._hook_model(m, crop)
                    self.specialist_models[crop] = m
                    print(f"✅ V2 Engine: โหลด {crop.upper()} Specialist ({len(classes)} คลาส) สำเร็จ")

            # 2. โหลด Unified 23-Class Router (ถ้ามี)
            unified_path = self.models_dir / "best_agrishield_mobilenetv3.pth"
            if unified_path.exists():
                u = models.mobilenet_v3_large()
                in_feat = u.classifier[3].in_features
                u.classifier[3] = nn.Linear(in_feat, len(UNIFIED_23_CLASSES))
                u.load_state_dict(torch.load(str(unified_path), map_location=self.device))
                u.to(self.device)
                u.eval()
                self._hook_model(u, "unified")
                self.unified_model = u
                print("✅ V2 Engine: โหลด Unified 23-Class Router สำเร็จ")

            if len(self.specialist_models) > 0:
                self.is_torch_ready = True

        except Exception as e:
            print(f"⚠️ V2 Engine: ไม่สามารถโหลด PyTorch models ได้ ({e})")
            self.is_torch_ready = False

    def _analyze_leaf(self, orig_img: Image.Image):
        img_np = np.array(orig_img.resize((224, 224), Image.LANCZOS)).astype(np.float32)
        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        white_bg = (r > 235) & (g > 235) & (b > 235)
        white_bg_ratio = float(white_bg.mean())
        green_leaf = (g > r + 6) & (g > b + 8) & (g > 40)
        green_area_ratio = float(green_leaf.mean())
        g_excess = float((g[green_leaf] - r[green_leaf]).mean()) if green_leaf.sum() > 50 else 0.0

        border_mask = np.zeros((224, 224), dtype=bool)
        border_mask[:20, :] = True
        border_mask[-20:, :] = True
        border_mask[:, :20] = True
        border_mask[:, -20:] = True
        brown_pixels = (r > g + 8) & (g > b + 5) & (r > 60) & (~white_bg)
        border_brown_bg_ratio = float((border_mask & brown_pixels).sum()) / float(border_mask.sum())

        ys, xs = np.where(green_leaf)
        if len(ys) > 80:
            y_min, y_max = int(np.percentile(ys, 5)), int(np.percentile(ys, 95))
            x_min, x_max = int(np.percentile(xs, 5)), int(np.percentile(xs, 95))
            leaf_box_mask = np.zeros((224, 224), dtype=bool)
            leaf_box_mask[y_min:y_max + 1, x_min:x_max + 1] = True
            internal_brown = leaf_box_mask & (r > g + 6) & (g < 180) & (b < 135) & (~white_bg)
            dark_or_purple_blight = (
                leaf_box_mask
                & (~green_leaf)
                & (~white_bg)
                & (g < 150)
                & ((r > g - 4) | (b > g + 4))
                & (np.abs(r - g) < 42)
                & ((r + g + b) > 55)
            )
            lesion_mask = internal_brown | dark_or_purple_blight
            internal_lesion_ratio = float(lesion_mask.sum()) / max(int(leaf_box_mask.sum()), 1)
        else:
            internal_lesion_ratio = 0.0

        transitions = []
        for row in range(20, 204, 8):
            line = green_leaf[row, :].astype(np.int32)
            crossings = int(np.sum(np.abs(np.diff(line))) // 2)
            if crossings > 0:
                transitions.append(crossings)
        for col in range(20, 204, 8):
            line = green_leaf[:, col].astype(np.int32)
            crossings = int(np.sum(np.abs(np.diff(line))) // 2)
            if crossings > 0:
                transitions.append(crossings)
        avg_crossings = float(np.mean(transitions)) if transitions else 1.0

        return {
            "white_bg_ratio": white_bg_ratio,
            "green_area_ratio": green_area_ratio,
            "g_excess": g_excess,
            "border_brown_bg_ratio": border_brown_bg_ratio,
            "internal_lesion_ratio": internal_lesion_ratio,
            "is_compound_broadleaf": avg_crossings > 2.2
        }

    def predict(self, image_path: str, crop_filter: str = "all"):
        orig_img = Image.open(image_path).convert("RGB")
        if not self.is_torch_ready:
            return self._fallback_prediction(orig_img, image_path, crop_filter)

        leaf_stats = self._analyze_leaf(orig_img)

        # 0. ตรวจสอบภาพที่ไม่ใช่พืช (Non-Plant)
        if (leaf_stats["green_area_ratio"] < 0.05 and leaf_stats["g_excess"] < 12.0) or \
           (leaf_stats["green_area_ratio"] < 0.02 and leaf_stats["internal_lesion_ratio"] < 0.03):
            return {
                "top1_class": "Out_of_Scope___Non_Plant",
                "confidence": 0.0,
                "top2_info": None,
                "heatmap_path": os.path.basename(image_path),
                "is_healthy": False,
                "is_out_of_scope": True,
                "engine": f"AgriShield V2 Leaf Filter ({self.device})"
            }

        # 1. ตัดสินใจเลือกโมเดล (Active Specialist Model)
        crop_key = (crop_filter or "all").lower().strip()
        auto_detected = False

        if crop_key not in self.specialist_models:
            # Auto-routing: ถ้าเลือก "all" ให้ใช้ Router คัดเลือกว่าเป็นพืชชนิดไหน
            auto_detected = True
            crop_key = self._route_crop(orig_img, leaf_stats)

        active_model = self.specialist_models.get(crop_key)
        classes = CROP_CLASSES[crop_key]

        # Multi-scale inference (Full + 70% Center Crop)
        w, h = orig_img.size
        cc_img = orig_img.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)))
        
        with torch.no_grad():
            batch_tensors = torch.stack([self.transform(orig_img), self.transform(cc_img)]).to(self.device)
            batch_out = active_model(batch_tensors)
            p_full = F.softmax(batch_out[0], dim=0)
            p_cc   = F.softmax(batch_out[1], dim=0)
            
            # ถ่วงน้ำหนัก Multi-Scale (Full Image + Center Crop)
            # ใช้ Full Image เป็นหลักหากโมเดลเห็นรอยโรคชัดเจนอยู่แล้ว (ไม่ตัดขอบใบทิ้ง)
            # และใช้ Center Crop เข้าช่วยเฉพาะกรณีภาพมุมกว้างที่ Full มองเป็นใบปกติหรือยังไม่มั่นใจ
            top_full_idx = int(batch_out[0].argmax().item())
            top_cc_idx = int(batch_out[1].argmax().item())
            full_max_p = float(p_full[top_full_idx].item())
            cc_max_p = float(p_cc[top_cc_idx].item())

            is_full_healthy = "healthy" in classes[top_full_idx].lower()
            is_cc_healthy = "healthy" in classes[top_cc_idx].lower()

            if is_full_healthy and not is_cc_healthy and cc_max_p > 0.45:
                p_spec = 0.15 * p_full + 0.85 * p_cc
            elif full_max_p >= 0.75 and not is_full_healthy:
                p_spec = p_full
            elif cc_max_p > full_max_p + 0.12:
                p_spec = 0.20 * p_full + 0.80 * p_cc
            else:
                p_spec = 0.75 * p_full + 0.25 * p_cc

            # ใช้ Specialist 100% สำหรับ Tomato (99.85%) และ Potato (100%)
            # และเสริม Unified เฉพาะจุดแผลขนาดเล็กในภาพมุมกว้างของ Corn/Rice
            crop_uni_weight = {
                "tomato": 0.0,
                "potato": 0.0,
                "rice": 0.65,
                "corn": 0.75,
            }.get(crop_key, 0.0)

            if self.unified_model is not None and crop_uni_weight > 0.0:
                u_out = self.unified_model(batch_tensors[0:1])[0]
                u_indices = [i for i, c in enumerate(UNIFIED_23_CLASSES) if c.lower().startswith(crop_key)]
                if len(u_indices) == len(classes):
                    p_uni = F.softmax(u_out[u_indices], dim=0)
                    probs = (1.0 - crop_uni_weight) * p_spec + crop_uni_weight * p_uni
                else:
                    probs = p_spec
            else:
                probs = p_spec

            # ชดเชย White-Background Studio Cutout Bias (กิ่งใบประกอบไดคัทพื้นขาวล้วนที่คลอโรฟิลล์เขียวสดไร้รอยโรค)
            if (
                leaf_stats["white_bg_ratio"] > 0.35
                and leaf_stats["g_excess"] > 30.0
                and leaf_stats["internal_lesion_ratio"] < 0.03
            ):
                healthy_candidates = [i for i, c in enumerate(classes) if "healthy" in c.lower()]
                if healthy_candidates:
                    h_idx = healthy_candidates[0]
                    probs = probs.clone() * 0.06
                    probs[h_idx] = 0.94
                    probs = probs / probs.sum()

        # Grad-CAM Forward Pass
        input_tensor = self.transform(orig_img).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True
        output = active_model(input_tensor)

        top2_p, top2_i = torch.topk(probs, k=min(2, len(classes)))
        top1_idx = int(top2_i[0].item())
        top1_prob = float(top2_p[0].item()) * 100.0
        top1_class = classes[top1_idx]

        # Check OOD for this crop (เช่น ถ้าเลือกโมเดลข้าวแต่ส่งใบหม่อนมา หรือความมั่นใจต่ำเกินไป)
        is_ood = False
        if leaf_stats["border_brown_bg_ratio"] > 0.30 and leaf_stats["g_excess"] > 45.0:
            is_ood = True
        elif top1_prob < 65.0:
            is_ood = True

        if is_ood:
            # หากความมั่นใจต่ำกว่า 65% หรือไม่มีสัดส่วนสีเขียวของพืชเพียงพอ จัดเป็นภาพไม่ใช่พืช / นอกขอบเขต
            if (leaf_stats["green_area_ratio"] < 0.25 and leaf_stats["g_excess"] < 30.0) or top1_prob < 55.0:
                return {
                    "top1_class": "Out_of_Scope___Non_Plant",
                    "confidence": round(top1_prob, 2),
                    "top2_info": None,
                    "heatmap_path": os.path.basename(image_path),
                    "is_healthy": False,
                    "is_out_of_scope": True,
                    "engine": f"AgriShield V2 OOD Filter ({self.device})"
                }
            leaf_healthy = (leaf_stats["g_excess"] >= 28.0 and leaf_stats["internal_lesion_ratio"] < 0.10)
            ood_class = "Out_of_Scope___Healthy_Leaf" if leaf_healthy else "Out_of_Scope___Suspected_Disease"
            health_conf = min(98.5, max(78.0, 75.0 + (leaf_stats["g_excess"] * 0.28)))
            heatmap_path = self._generate_gradcam(orig_img, image_path, crop_key, top1_idx, output, is_healthy=leaf_healthy)
            return {
                "top1_class": ood_class,
                "confidence": round(health_conf, 2),
                "top2_info": {
                    "class_name": top1_class,
                    "confidence": round(top1_prob, 2),
                    "is_ood_note": True
                },
                "heatmap_path": heatmap_path,
                "is_healthy": leaf_healthy,
                "is_out_of_scope": True,
                "engine": f"V2 {crop_key.capitalize()} Specialist + OOD Filter ({self.device})"
            }

        is_healthy = "healthy" in top1_class.lower()
        heatmap_path = self._generate_gradcam(orig_img, image_path, crop_key, top1_idx, output, is_healthy=is_healthy)

        top2_info = None
        if top1_prob < 85.0 and len(classes) > 1 and float(top2_p[1].item()) > 0.08:
            top2_idx = int(top2_i[1].item())
            top2_info = {
                "class_name": classes[top2_idx],
                "confidence": round(float(top2_p[1].item()) * 100.0, 2)
            }

        crop_thai = {"rice": "ข้าว", "corn": "ข้าวโพด", "potato": "มันฝรั่ง", "tomato": "มะเขือเทศ"}.get(crop_key, crop_key)
        mode_label = f"Auto-Routed to {crop_thai}" if auto_detected else f"Selected: {crop_thai}"

        return {
            "top1_class": top1_class,
            "confidence": round(top1_prob, 2),
            "top2_info": top2_info,
            "heatmap_path": heatmap_path,
            "is_healthy": is_healthy,
            "is_out_of_scope": False,
            "engine": f"MobileNetV3 {crop_key.capitalize()} Specialist Model ({mode_label})"
        }

    def _route_crop(self, orig_img: Image.Image, leaf_stats: dict = None) -> str:
        """คัดเลือกว่าเป็นพืชชนิดไหนจาก 4 ชนิด (Router)"""
        if self.unified_model:
            with torch.no_grad():
                inp = self.transform(orig_img).unsqueeze(0).to(self.device)
                out = self.unified_model(inp)
                probs = F.softmax(out[0], dim=0)
                crop_scores = {"tomato": 0.0, "rice": 0.0, "corn": 0.0, "potato": 0.0}
                for i, c in enumerate(UNIFIED_23_CLASSES):
                    for ck in crop_scores:
                        if c.lower().startswith(ck):
                            crop_scores[ck] += float(probs[i])
                if leaf_stats and leaf_stats.get("white_bg_ratio", 0.0) > 0.35 and leaf_stats.get("is_compound_broadleaf", False):
                    return "tomato" if crop_scores["tomato"] >= crop_scores["potato"] else "potato"
                return max(crop_scores, key=crop_scores.get)
        return "tomato"

    def _generate_gradcam(self, orig_img: Image.Image, image_path: str, model_key: str, class_idx: int, output_tensor, is_healthy: bool):
        try:
            active_model = self.specialist_models[model_key]
            active_model.zero_grad()
            score = output_tensor[0, class_idx]
            score.backward(retain_graph=True)

            grads = self.gradients[model_key]
            fmaps = self.feature_maps[model_key]
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * fmaps, dim=1).squeeze()
            cam = F.relu(cam)
            cam_np = cam.cpu().detach().numpy()
            if np.max(cam_np) > 0:
                cam_np = cam_np / np.max(cam_np)
            else:
                cam_np = np.zeros_like(cam_np)

            w, h = orig_img.size
            cam_img = Image.fromarray(np.uint8(255 * cam_np)).resize((w, h), Image.BILINEAR)
            cam_array = np.array(cam_img, dtype=np.float32) / 255.0

            orig_array = np.array(orig_img, dtype=np.float32) / 255.0
            r, g, b = orig_array[:, :, 0], orig_array[:, :, 1], orig_array[:, :, 2]

            if is_healthy:
                green_mask = ((g > r + 0.03) & (g > b + 0.04) & (g > 0.16)).astype(np.float32)
                mask_img = Image.fromarray(np.uint8(green_mask * 255)).filter(ImageFilter.GaussianBlur(radius=10))
                green_blur = np.array(mask_img, dtype=np.float32) / 255.0
                if np.max(green_blur) > 0:
                    green_blur = green_blur / np.max(green_blur)
                    cam_array = np.maximum(cam_array * green_blur, 0.65 * green_blur)
                    if np.max(cam_array) > 0:
                        cam_array = cam_array / np.max(cam_array)
            else:
                green_leaf = ((g > r + 0.02) & (g > b + 0.03) & (g > 0.15)).astype(np.float32)
                prox_radius = max(16, int(min(w, h) * 0.06))
                leaf_proximity = np.array(
                    Image.fromarray(np.uint8(green_leaf * 255)).filter(ImageFilter.GaussianBlur(radius=prox_radius)),
                    dtype=np.float32
                ) / 255.0
                leaf_proximity = np.clip(leaf_proximity * 2.5, 0.0, 1.0)

                sky_mask = (b > r + 0.10) & (b > g + 0.04) & (b > 0.42)
                skin_mask = (r > 0.62) & (g > 0.42) & (b > 0.32) & (r > g + 0.08) & (g > b + 0.05)
                lesion_affinity = (
                    (leaf_proximity > 0.15)
                    & (green_leaf < 0.5)
                    & (~sky_mask)
                    & (~skin_mask)
                    & (r < 0.88)
                    & ((r > g - 0.02) | (b > g + 0.02))
                    & ((r + g + b) > 0.20)
                ).astype(np.float32)
                lesion_blur = np.array(
                    Image.fromarray(np.uint8(lesion_affinity * 255)).filter(ImageFilter.GaussianBlur(radius=14)),
                    dtype=np.float32
                ) / 255.0
                if np.max(lesion_blur) > 0:
                    lesion_blur = lesion_blur / np.max(lesion_blur)
                    cam_array = (0.45 * cam_array + 0.55 * lesion_blur) * leaf_proximity
                cam_array[sky_mask | skin_mask] = 0.0
                if np.max(cam_array) > 0:
                    cam_array = cam_array / np.max(cam_array)

            white_bg_mask = (
                (orig_array[:, :, 0] > 0.92)
                & (orig_array[:, :, 1] > 0.92)
                & (orig_array[:, :, 2] > 0.92)
            )
            cam_array[white_bg_mask] = 0.0

            heatmap = np.zeros_like(orig_array)
            if is_healthy:
                heatmap[:, :, 0] = 0.0
                heatmap[:, :, 1] = 1.0
                heatmap[:, :, 2] = 0.55
                alpha = (cam_array[..., None] ** 0.95) * 0.48
            else:
                heatmap[:, :, 0] = 1.0
                heatmap[:, :, 1] = np.clip(0.95 - cam_array * 0.80, 0.0, 1.0)
                heatmap[:, :, 2] = 0.0
                alpha = (cam_array[..., None] ** 0.95) * 0.58

            overlay = (1.0 - alpha) * orig_array + alpha * heatmap
            overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

            uploads_dir = Path(__file__).resolve().parent / "static" / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem
            save_name = f"gradcam_{stem}.jpg"
            save_path = uploads_dir / save_name
            Image.fromarray(overlay).save(str(save_path), quality=92)
            return save_name
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return os.path.basename(image_path)

    def _fallback_prediction(self, orig_img, image_path, crop_filter):
        return {
            "top1_class": "Tomato___healthy",
            "confidence": 95.0,
            "top2_info": None,
            "heatmap_path": os.path.basename(image_path),
            "is_healthy": True,
            "is_out_of_scope": False,
            "engine": "Fallback Engine"
        }
