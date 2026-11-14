# ==============================================================================
# 🚀 AgriShield AI — FastAPI Application Server
# ==============================================================================
import os
import sys
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# นำเข้าโมดูลโมเดลและฐานข้อมูลคำแนะนำ
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from model_engine import AgriShieldModelEngine
from recommendation_db import get_recommendation, RECOMMENDATIONS

# ตรวจสอบตำแหน่งโมเดล .pth
MODEL_PATHS = [
    BASE_DIR / "best_agrishield_mobilenetv3.pth",
    BASE_DIR.parent / "best_agrishield_mobilenetv3.pth"
]
active_model_path = None
for p in MODEL_PATHS:
    if p.exists():
        active_model_path = str(p)
        break

if not active_model_path:
    active_model_path = str(BASE_DIR.parent / "best_agrishield_mobilenetv3.pth")

print(f"กำหนดเส้นทางโมเดล: {active_model_path}")
engine = AgriShieldModelEngine(model_path=active_model_path)

# เริ่มต้นสร้างแอป FastAPI
app = FastAPI(
    title="AgriShield AI",
    description="ระบบวินิจฉัยโรคพืช 23 คลาส และให้คำแนะนำชีวภัณฑ์ที่เป็นมิตรต่อสิ่งแวดล้อม (SDG 12)",
    version="1.0.0"
)

# ผูก Static Files และ Templates
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """หน้าแรกสำหรับอัปโหลดภาพหรือถ่ายภาพสด"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    crop_filter: str = Form("all")
):
    """รับไฟล์ภาพ วินิจฉัยโรค และส่งคืนหน้าผลลัพธ์ HTML"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="กรุณาเลือกไฟล์ภาพ")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    safe_filename = f"{uuid.uuid4().hex[:12]}{ext}"
    saved_path = UPLOAD_DIR / safe_filename

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    diagnosis = engine.predict(str(saved_path), crop_filter=crop_filter)
    top1_class = diagnosis["top1_class"]
    confidence = diagnosis["confidence"]
    top2_info = diagnosis.get("top2_info")
    heatmap_filename = diagnosis.get("heatmap_path", safe_filename)

    recommendation = get_recommendation(top1_class)
    is_healthy = diagnosis.get("is_healthy", "healthy" in top1_class.lower())
    is_out_of_scope = diagnosis.get("is_out_of_scope", top1_class.startswith("Out_of_Scope"))

    if top1_class == "Out_of_Scope___Non_Plant":
        gradcam_caption = "ไม่แสดงผล Grad-CAM เนื่องจากระบบตรวจไม่พบเนื้อใบพืชในภาพที่อัปโหลด"
        status_label = "ตรวจไม่พบใบพืชในภาพ (Non-Plant Image)"
        status_badge_style = "background:#f1f5f9; color:#334155; border-color:#cbd5e1;"
    elif is_out_of_scope:
        gradcam_caption = (
            "พื้นที่สีเขียว-ฟ้า คือบริเวณเนื้อใบสุขภาพดีที่ระบบตรวจพบว่าคลอโรฟิลล์สมบูรณ์ไร้รอยโรค"
            if is_healthy
            else "พื้นที่สีส้ม-แดง คือตำแหน่งร่องรอยความผิดปกติบนแผ่นใบที่ระบบตรวจพบ"
        )
        status_label = (
            "พืชนอกขอบเขต 4 ชนิดหลัก · ใบสมบูรณ์แข็งแรง"
            if is_healthy
            else "พืชนอกขอบเขต 4 ชนิดหลัก · พบความผิดปกติ"
        )
        status_badge_style = (
            "background:#e0f2fe; color:#075985; border-color:#bae6fd;"
            if is_healthy
            else "background:#fffbeb; color:#92400e; border-color:#fde68a;"
        )
    else:
        gradcam_caption = (
            "พื้นที่สีเขียว-ฟ้า คือบริเวณเนื้อใบสุขภาพดีที่ระบบตรวจพบว่าคลอโรฟิลล์สมบูรณ์ไร้รอยโรค"
            if is_healthy
            else "พื้นที่สีส้ม-แดง คือตำแหน่งรอยโรคที่โมเดลตรวจพบและใช้ตัดสินใจ"
        )
        status_label = "ใบพืชสุขภาพดี (Healthy)" if is_healthy else "ตรวจพบรอยโรคพืช (Disease Detected)"
        status_badge_style = (
            "background:#ecfdf5; color:#065f46; border-color:#a7f3d0;"
            if is_healthy
            else "background:#fffbeb; color:#92400e; border-color:#fde68a;"
        )

    if top2_info:
        top2_rec = get_recommendation(top2_info["class_name"])
        top2_info["thai_name"] = top2_rec.get("thai_name", top2_info["class_name"])

    return templates.TemplateResponse("result.html", {
        "request": request,
        "top1_class": top1_class,
        "confidence": confidence,
        "top2_info": top2_info,
        "is_healthy": is_healthy,
        "orig_image": safe_filename,
        "heatmap_image": heatmap_filename,
        "gradcam_caption": gradcam_caption,
        "status_label": status_label,
        "status_badge_style": status_badge_style,
        "recommendation": recommendation,
        "engine_info": diagnosis.get("engine", "MobileNetV3")
    })


@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    crop_filter: str = Form("all")
):
    """REST API Endpoint สำหรับเชื่อมต่อกับ Mobile App หรือ External Service"""
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    safe_filename = f"api_{uuid.uuid4().hex[:12]}{ext}"
    saved_path = UPLOAD_DIR / safe_filename

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    diagnosis = engine.predict(str(saved_path), crop_filter=crop_filter)
    top1_class = diagnosis["top1_class"]
    recommendation = get_recommendation(top1_class)

    return {
        "status": "success",
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "original_image_url": f"/static/uploads/{safe_filename}",
        "heatmap_image_url": f"/static/uploads/{diagnosis.get('heatmap_path', safe_filename)}"
    }


@app.get("/api/classes")
async def get_all_classes():
    """ส่งคืนรายชื่อทั้ง 23 คลาสและพืชเป้าหมาย"""
    return {
        "total_classes": len(RECOMMENDATIONS),
        "crops": ["Corn (ข้าวโพด)", "Potato (มันฝรั่ง)", "Rice (ข้าว)", "Tomato (มะเขือเทศ)"],
        "classes": list(RECOMMENDATIONS.keys())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
