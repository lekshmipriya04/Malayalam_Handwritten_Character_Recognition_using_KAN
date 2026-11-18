import base64
import io
from pathlib import Path
from zipfile import BadZipFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app import (
    IMAGE_EXTENSIONS,
    MODEL_PATH,
    load_model,
    predict,
    predict_zip,
    preprocess_image,
)


ROOT_DIR = Path(__file__).resolve().parent
WEB_DIST_DIR = ROOT_DIR / "web" / "dist"

app = FastAPI(title="Malayalam Character Recognition")


def get_model():
    checkpoint_stat = MODEL_PATH.stat()
    return load_model(checkpoint_stat.st_mtime_ns, checkpoint_stat.st_size)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict_upload(
    file: UploadFile = File(...),
    already_preprocessed: bool = Form(False),
):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if extension != ".zip" and extension not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a supported image or a ZIP archive.",
        )

    try:
        model, classes, device = get_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    if extension == ".zip":
        try:
            results = predict_zip(
                contents,
                model,
                classes,
                device,
                already_preprocessed=already_preprocessed,
            )
        except (BadZipFile, OSError, ValueError, RuntimeError) as error:
            raise HTTPException(
                status_code=400, detail=f"Could not process this ZIP file: {error}"
            ) from error
        return {"type": "zip", "filename": filename, "results": results}

    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.load()
            input_tensor, preview, preprocessing_mode = preprocess_image(
                image, already_preprocessed=already_preprocessed
            )
        predictions = predict(model, classes, device, input_tensor)
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=400, detail=f"Could not process this image: {error}"
        ) from error

    preview_buffer = io.BytesIO()
    preview.save(preview_buffer, format="PNG")
    preview_data = base64.b64encode(preview_buffer.getvalue()).decode("ascii")
    return {
        "type": "image",
        "filename": filename,
        "preprocessing_mode": preprocessing_mode,
        "preview": f"data:image/png;base64,{preview_data}",
        "predictions": [
            {"character": character, "confidence": confidence}
            for character, confidence in predictions
        ],
    }


if WEB_DIST_DIR.is_dir():
    assets_dir = WEB_DIST_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{path:path}")
    def serve_react(path: str):
        requested_file = (WEB_DIST_DIR / path).resolve()
        if requested_file.is_relative_to(WEB_DIST_DIR.resolve()) and requested_file.is_file():
            return FileResponse(requested_file)
        return FileResponse(WEB_DIST_DIR / "index.html")
