# main.py — ส่งผลลัพธ์ + รูป preview เป็น base64
import io, os, tempfile, base64
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # สำคัญ: ป้องกัน backend มีจอ
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ใช้ตัววิเคราะห์และฟังก์ชันวาดจาก face_analyze.py
from .face_analyze import CombinedFaceAnalyzer, visualize_plt

app = FastAPI(title="Face Pose & Emotion API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_analyzer = None
_startup_error = None
try:
    _analyzer = CombinedFaceAnalyzer(staticMode=True, maxFaces=5, use_emotions=True)
except Exception as e:
    _startup_error = str(e)

@app.get("/health")
async def health():
    return {
        "status": "healthy" if _analyzer and not _startup_error else "error",
        "model_loaded": bool(_analyzer and _analyzer.use_emotions),
        "error": _startup_error,
    }

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...), preview: bool = True):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    if _analyzer is None:
        raise HTTPException(status_code=500, detail=f"Analyzer not ready: {_startup_error or 'unknown error'}")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)
    image_bgr = image_np[:, :, ::-1].copy()

    (result_img, landmarks_list, angle_R_list, angle_L_list,
     pose_list, pan_list, emotions_list, conf_list) = _analyzer.analyze_face(image_bgr, draw=False)

    faces = []
    for i, (pose, pan, emo, conf) in enumerate(zip(pose_list, pan_list, emotions_list, conf_list)):
        box = None
        if i < len(landmarks_list):
            x_coords = [p[0] for p in landmarks_list[i]]
            y_coords = [p[1] for p in landmarks_list[i]]
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
            box = {
                "x": x_min,
                "y": y_min,
                "w": x_max - x_min,
                "h": y_max - y_min
            }

        faces.append({
            "pose": pose,
            "pan": float(np.round(pan, 1)),
            "emotion": emo,
            "confidence": float(conf),
            "box": box
        })


    first_emotion = faces[0]["emotion"] if faces else "N/A"
    first_conf = faces[0]["confidence"] if faces else 0.0

    resp = {
        "prediction": first_emotion,
        "confidence": first_conf,
        "filename": file.filename,
        "faces": faces, 
        "face_count": len(faces),
        "image_width": image_np.shape[1],
        "image_height": image_np.shape[0]
    }

    if preview:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            out_path = visualize_plt(
                image_bgr, landmarks_list, angle_R_list, angle_L_list,
                pose_list, emotions_list, pan_list, save_path=tmp.name, show=False
            )
        with open(out_path, "rb") as f:
            resp["preview_base64"] = base64.b64encode(f.read()).decode("utf-8")
            resp["preview_mime"] = "image/jpeg"  # หรือ image/png แล้วแต่ format
        os.unlink(out_path)

    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("face_analyze.main:app", host="0.0.0.0", port=8000, reload=True)
