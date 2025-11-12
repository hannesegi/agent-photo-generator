
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.exceptions import RequestValidationError
from loguru import logger
from datetime import datetime
import traceback
import os
import sys
import time
from typing import Optional, Dict, Any, List

path_this = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(os.path.join(path_this, '..'))
path_root = os.path.dirname(os.path.join(path_this, '../..'))
sys.path.extend([path_this, path_project, path_root])

from tools.tools_generate_i2i import SDImg2Img   

app = FastAPI(
    title="Image2Image API",
    description="API for Img2Img Photo",
    version="1.0.0"
)


class Img2ImgRequest(BaseModel):
    images_b64: List[str] = Field(..., description="List of base-64 encoded init images")
    prompt: str = Field(..., description="Positive prompt")
    negative_prompt: str = Field("", description="Negative prompt")
    steps: Optional[int] = Field(30, ge=1, le=150, description="Sampling steps")
    cfg_scale: Optional[float] = Field(7.0, ge=1.0, le=30.0, description="CFG scale")
    denoising_strength: Optional[float] = Field(0.75, ge=0.0, le=1.0, description="Denoising strength")
    sampler_name: Optional[str] = Field("DPM++ 2M Karras", description="Sampler name")
    output_dir: Optional[str] = Field("result", description="Folder to save outputs")

class APIResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    elapsed_time: Optional[float]

# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Exception handlers
# -------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content=APIResponse(
            status="error",
            error=f"Invalid request parameters: {exc.errors()}",
            data=None,
            elapsed_time=None
        ).dict()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content=APIResponse(
            status="error",
            error=str(exc),
            data=None,
            elapsed_time=None
        ).dict()
    )

# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "img2img-fastapi"}


@app.post("/img2img", response_model=APIResponse)
def img2img_endpoint(payload: Img2ImgRequest = Body(...)):
    """
    Generate images from base64 init images using Stable Diffusion img2img.
    Returns metadata containing base64, local file path, and elapsed time.
    """
    start = time.time()
    try:
        sd = SDImg2Img(
            images_b64=payload.images_b64,
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            steps=payload.steps,
            cfg_scale=payload.cfg_scale,
            denoising_strength=payload.denoising_strength,
            sampler_name=payload.sampler_name,
            output_dir=payload.output_dir
        )

        metadata = sd.generate_and_save()
        elapsed = time.time() - start

        return APIResponse(
            status="success",
            data={"images": metadata},
            error=None,
            elapsed_time=elapsed
        )

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Img2Img error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=APIResponse(
                status="error",
                data=None,
                error=str(e),
                elapsed_time=elapsed
            ).dict()
        )


@app.get("/result/{filename}")
def get_result_file(filename: str):
    """
    Serve generated images from the default result folder.
    Only allow png/jpg/jpeg extensions.
    """
    safe_filename = os.path.basename(filename)
    file_path = os.path.join("result", safe_filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_service_img2img:app",
        host="0.0.0.0",
        port=7028,
        reload=False
    )