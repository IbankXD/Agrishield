# ==============================================================================
# 🌐 AgriShield AI — Lightweight Standalone Server (Zero External Dependencies)
# รองรับทั้ง Python 3.10 - 3.14+ พร้อมระบบ Auto-Reload โมเดลและตรวจจับพืชนอกขอบเขต
# ==============================================================================
import os
import sys

# บังคับสลับไปใช้ Python 3.14 ที่มี PyTorch ติดตั้งอยู่โดยอัตโนมัติ (หากผู้ใช้เผลอรันด้วย Python ตัวอื่น)
_PY314_EXE = r"C:\Users\door_\AppData\Local\Programs\Python\Python314\python.exe"
_PY314_SITE = r"C:\Users\door_\AppData\Local\Programs\Python\Python314\Lib\site-packages"
_PY314_USER_SITE = r"C:\Users\door_\AppData\Roaming\Python\Python314\site-packages"

for _sp in (_PY314_SITE, _PY314_USER_SITE):
    if os.path.exists(_sp) and _sp not in sys.path:
        sys.path.insert(0, _sp)

if (
    os.path.exists(_PY314_EXE)
    and os.path.normcase(sys.executable) != os.path.normcase(_PY314_EXE)
    and os.environ.get("AGRISHIELD_REEXEC") != "1"
):
    import subprocess
    env = os.environ.copy()
    env["AGRISHIELD_REEXEC"] = "1"
    sys.exit(subprocess.call([_PY314_EXE] + sys.argv, env=env))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import uuid
import json
import email
import mimetypes
import importlib
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import model_engine
import recommendation_db

# ค้นหาโมเดล .pth
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

print(f"[AgriShield Server] Python: {sys.executable}")
print(f"[AgriShield Server] โมเดลเป้าหมาย: {active_model_path}")
engine = model_engine.AgriShieldModelEngine(model_path=active_model_path)
_last_engine_mtime = os.path.getmtime(BASE_DIR / "model_engine.py")
_last_db_mtime = os.path.getmtime(BASE_DIR / "recommendation_db.py")

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_active_engine():
    """ตรวจสอบและรีโหลด model_engine / recommendation_db อัตโนมัติหากไฟล์ถูกอัปเดต"""
    global engine, _last_engine_mtime, _last_db_mtime
    try:
        cur_db_mtime = os.path.getmtime(BASE_DIR / "recommendation_db.py")
        if cur_db_mtime != _last_db_mtime:
            importlib.reload(recommendation_db)
            _last_db_mtime = cur_db_mtime

        cur_eng_mtime = os.path.getmtime(BASE_DIR / "model_engine.py")
        if cur_eng_mtime != _last_engine_mtime or not getattr(engine, "is_torch_ready", False):
            importlib.reload(model_engine)
            engine = model_engine.AgriShieldModelEngine(model_path=active_model_path)
            _last_engine_mtime = cur_eng_mtime
    except Exception as e:
        print(f"[AgriShield Server] Auto-reload warning: {e}")
    return engine


class AgriShieldHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.serve_file(BASE_DIR / "templates" / "index.html", "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            file_path = BASE_DIR / "static" / rel_path
            if file_path.exists() and file_path.is_file():
                mime, _ = mimetypes.guess_type(str(file_path))
                self.serve_file(file_path, mime or "application/octet-stream")
            else:
                self.send_error(404, "File not found")
        elif path == "/api/classes":
            payload = {
                "total_classes": len(recommendation_db.RECOMMENDATIONS),
                "crops": ["Corn", "Potato", "Rice", "Tomato"],
                "classes": list(recommendation_db.RECOMMENDATIONS.keys())
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/predict" or path == "/api/predict":
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self.send_error(400, "Bad Request: Expected multipart/form-data")
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)

            header_prefix = f"Content-Type: {content_type}\r\n\r\n".encode("latin-1")
            msg = email.message_from_bytes(header_prefix + body_bytes)

            file_bytes = None
            filename = None
            crop_filter = "all"

            for part in msg.walk():
                disp = part.get("Content-Disposition", "")
                fn = part.get_filename()
                if fn:
                    filename = fn
                    file_bytes = part.get_payload(decode=True)
                elif 'name="file"' in disp and not file_bytes:
                    file_bytes = part.get_payload(decode=True)
                    filename = "snapshot.jpg"
                elif 'name="crop_filter"' in disp:
                    payload_bytes = part.get_payload(decode=True)
                    if payload_bytes:
                        crop_filter = payload_bytes.decode("utf-8", errors="ignore").strip() or "all"

            if not file_bytes:
                self.send_error(400, "No file payload received")
                return

            ext = os.path.splitext(filename)[1].lower() if filename else ".jpg"
            if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                ext = ".jpg"

            safe_filename = f"{uuid.uuid4().hex[:12]}{ext}"
            saved_path = UPLOAD_DIR / safe_filename

            with open(saved_path, "wb") as f:
                f.write(file_bytes)

            # วินิจฉัยด้วย Engine ปัจจุบัน
            active_eng = get_active_engine()
            diagnosis = active_eng.predict(str(saved_path), crop_filter=crop_filter)
            top1_class = diagnosis["top1_class"]
            confidence = diagnosis["confidence"]
            top2_info = diagnosis.get("top2_info")
            heatmap_filename = diagnosis.get("heatmap_path", safe_filename)
            is_healthy = diagnosis.get("is_healthy", "healthy" in top1_class.lower())
            is_out_of_scope = diagnosis.get("is_out_of_scope", top1_class.startswith("Out_of_Scope"))
            recommendation = recommendation_db.get_recommendation(top1_class)

            if path == "/api/predict":
                resp_data = {
                    "status": "success",
                    "diagnosis": diagnosis,
                    "recommendation": recommendation,
                    "original_image": f"/static/uploads/{safe_filename}",
                    "heatmap_image": f"/static/uploads/{heatmap_filename}"
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(resp_data, ensure_ascii=False).encode("utf-8"))
            else:
                with open(BASE_DIR / "templates" / "result.html", "r", encoding="utf-8") as f:
                    html_content = f.read()

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

                html_content = html_content.replace("{{ recommendation.thai_name }}", recommendation.get("thai_name", ""))
                html_content = html_content.replace("{{ recommendation.crop }}", recommendation.get("crop", ""))
                html_content = html_content.replace("{{ top1_class }}", top1_class)
                html_content = html_content.replace("{{ confidence }}", str(confidence))
                html_content = html_content.replace("{{ orig_image }}", safe_filename)
                html_content = html_content.replace("{{ heatmap_image }}", heatmap_filename)
                html_content = html_content.replace("{{ gradcam_caption }}", gradcam_caption)
                html_content = html_content.replace("{{ status_label }}", status_label)
                html_content = html_content.replace("{{ status_badge_style }}", status_badge_style)
                html_content = html_content.replace("{{ recommendation.cause }}", recommendation.get("cause", ""))
                html_content = html_content.replace("{{ recommendation.symptoms }}", recommendation.get("symptoms", ""))
                html_content = html_content.replace("{{ recommendation.biocontrol_treatment }}", recommendation.get("biocontrol_treatment", ""))
                html_content = html_content.replace("{{ recommendation.natural_management }}", recommendation.get("natural_management", ""))
                html_content = html_content.replace("{{ recommendation.sdg_impact }}", recommendation.get("sdg_impact", ""))

                if top2_info:
                    top2_rec = recommendation_db.get_recommendation(top2_info["class_name"])
                    top2_thai = top2_rec.get("thai_name", top2_info["class_name"])
                    if top2_info.get("is_ood_note"):
                        ood_leaf_desc = (
                            "สำหรับภาพนี้ระบบพบว่าเนื้อใบมีคลอโรฟิลล์สีเขียวสมบูรณ์แข็งแรง แต่รูปทรงใบอยู่นอกขอบเขตพืชทั้ง 4 ชนิด"
                            if is_healthy
                            else "สำหรับภาพนี้ระบบตรวจพบร่องรอยความผิดปกติบนแผ่นใบ แต่รูปทรงสัณฐานวิทยาของใบอยู่นอกขอบเขตพืชทั้ง 4 ชนิด"
                        )
                        top2_block = f"""
                        <div role="alert" class="mt-5 p-4 rounded-xl bg-sky-50 text-sky-950 flex items-start gap-3.5 text-sm">
                            <svg class="w-5 h-5 text-sky-700 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="12" y1="16" x2="12" y2="12"></line>
                                <line x1="12" y1="8" x2="12.01" y2="8"></line>
                            </svg>
                            <div>
                                <strong class="font-semibold text-sky-950">การคัดกรองพืชนอกขอบเขต (Out-of-Scope Crop Detection)</strong>
                                <p class="text-xs text-sky-900 mt-1 leading-relaxed">
                                    โมเดล AgriShield AI ฝึกสอนเฉพาะพืชเศรษฐกิจ 4 ชนิดหลัก (ข้าว, มะเขือเทศ, ข้าวโพด, มันฝรั่ง — 23 คลาส) {ood_leaf_desc} (หากบังคับจำแนกใน 23 คลาส โมเดลให้ความใกล้เคียงสูงสุดเพียง <span class="tabular-nums font-semibold">{top2_info['confidence']}%</span>)
                                </p>
                            </div>
                        </div>
                        """
                    elif top2_info.get("is_mixed_blight_note"):
                        top2_block = f"""
                        <div role="alert" class="mt-5 p-4 rounded-xl bg-amber-50 text-amber-950 flex items-start gap-3.5 text-sm">
                            <svg class="w-5 h-5 text-amber-700 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                                <line x1="12" y1="9" x2="12" y2="13"></line>
                                <line x1="12" y1="17" x2="12.01" y2="17"></line>
                            </svg>
                            <div>
                                <strong class="font-semibold text-amber-950">ข้อควรระวังในการวินิจฉัยแยกโรค (Differential Diagnosis — Mixed Solanaceae Blight)</strong>
                                <p class="text-xs text-amber-900 mt-1 leading-relaxed">
                                    ภาพแปลงปลูกจริงพบรอยโรครุนแรงหลายลักษณะร่วมกันบนใบพืชตระกูลมะเขือเทศ–มันฝรั่ง (Solanaceae): นอกจากแผลวงแหวนซ้อนกันของโรคใบไหม้เช้า (Early Blight) แล้ว ยังพบร่องรอยแผลไหม้ชื้นและเส้นใยสีขาวที่คล้ายคลึงกับ
                                    <span class="font-semibold underline">{top2_thai} (Potato / Tomato Late Blight — Phytophthora infestans)</span> แนะนำให้ใช้ชีวภัณฑ์ทั้งไตรโคเดอร์มาและบาซิลลัส ซับทิลิส (BS) ควบคู่กัน
                                </p>
                            </div>
                        </div>
                        """
                    else:
                        top2_block = f"""
                        <div role="alert" class="mt-5 p-4 rounded-xl bg-amber-50 text-amber-950 flex items-start gap-3.5 text-sm">
                            <svg class="w-5 h-5 text-amber-700 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                                <line x1="12" y1="9" x2="12" y2="13"></line>
                                <line x1="12" y1="17" x2="12.01" y2="17"></line>
                            </svg>
                            <div>
                                <strong class="font-semibold text-amber-950">ข้อควรระวังในการวินิจฉัยแยกโรค (Differential Diagnosis Advisory)</strong>
                                <p class="text-xs text-amber-900 mt-1 leading-relaxed">
                                    ความมั่นใจของโมเดลต่ำกว่าเกณฑ์มาตรฐาน 85% เนื่องจากลักษณะแผลมีความคล้ายคลึงกัน ระบบตรวจพบความเป็นไปได้อันดับที่สองคือ
                                    <span class="font-semibold underline">{top2_thai} ({top2_info['class_name']})</span> ที่ระดับความมั่นใจ <span class="tabular-nums font-semibold">{top2_info['confidence']}%</span> แนะนำให้เปรียบเทียบลักษณะอาการร่วมกับตารางด้านล่าง
                                </p>
                            </div>
                        </div>
                        """
                    start_tag = "{% if top2_info %}"
                    end_tag = "{% endif %}"
                    if start_tag in html_content and end_tag in html_content:
                        part1 = html_content.split(start_tag, 1)[0]
                        part2 = html_content.rsplit(end_tag, 1)[1]
                        html_content = part1 + top2_block + part2
                else:
                    start_tag = "{% if top2_info %}"
                    end_tag = "{% endif %}"
                    if start_tag in html_content and end_tag in html_content:
                        part1 = html_content.split(start_tag, 1)[0]
                        part2 = html_content.rsplit(end_tag, 1)[1]
                        html_content = part1 + part2

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html_content.encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def serve_file(self, file_path: Path, content_type: str):
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def log_message(self, format, *args):
        pass


def run(port=7000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, AgriShieldHandler)
    print(f"[AgriShield Server] พร้อมใช้งานที่ http://127.0.0.1:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    port = 7000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run(port)
