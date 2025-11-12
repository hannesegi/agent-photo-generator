from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from loguru import logger
import traceback
import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import  Any
import multiprocessing
import asyncio
from typing import  Any


path_this = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(os.path.join(path_this, '..'))
path_root = os.path.dirname(os.path.join(path_this, '../..'))
sys.path.extend([path_this, path_project, path_root])

from main_photo_generatort2i import ImageGenAgent

app = FastAPI(
    title="Text2Image Generator Agent API",
    description="API for create photo profil from Prompting",
    version="1.0.0"
)

PROFILE_EXAMPLE = {
    "prompt": "buatkan saya poto profil pria, usia muda ganteng berpakaian formal",
}

class PromptData(BaseModel):
    prompt: str = Field(..., example="buatkan saya poto profil pria, usia muda ganteng berpakaian formal")


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Request validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(request: Request, exc: ResponseValidationError):
    logger.error(f"Response validation error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error_message": str(exc),
            "elapsed_time": 0.0,
            "status": "error"
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": str(type(exc).__name__),
            "status": "error"
        },
    )

agent = None
executor = None

@app.on_event("startup")
async def startup_event():
    global agent, executor
    logger.info("Initializing ImageGenAgent...")
    agent = ImageGenAgent()
    executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    global executor
    if executor:
        executor.shutdown()
    logger.info("Application shutdown complete")

@app.post("/generate-photo-profile/", summary="Generate Photo Profile")
async def generate_photo_profile(request: PromptData):
    input_data = request
    try:
        logger.info(f"process generate from : {input_data}")
        
        # Run the image generation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        process_generate = await loop.run_in_executor(
            executor, 
            agent.process_generate_image, 
            input_data
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": process_generate,
                "message": "Photo generated successfully"
            }
        )
            
    except Exception as e:
        logger.error(f"Error generating photo: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Try to send Telegram notification if implemented in your agent
        try:
            if hasattr(agent, 'send_telegram_notification'):
                agent.send_telegram_notification(f"Error generating photo: {str(e)}")
        except Exception as tg_error:
            logger.warning(f"Failed to send error Telegram notification: {str(tg_error)}")
        
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7020)