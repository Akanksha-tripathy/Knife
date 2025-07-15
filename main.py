# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
from detect import detect_knives_yolo  # Make sure detect.py is in the same folder

app = FastAPI()

# Allow cross-origin requests (for use with HTML frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        # ✅ Check for valid video formats
        if not file.filename.endswith((".mp4", ".avi", ".mov")):
            raise HTTPException(status_code=415, detail="Unsupported video format.")

        # ✅ Save uploaded video to a folder
        os.makedirs("uploads", exist_ok=True)
        save_path = os.path.join("uploads", file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ✅ Run knife detection
        results = detect_knives_yolo(save_path)

        return JSONResponse(content={"detections": results})

    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ✅ Start the FastAPI server when run directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
