from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from model import Model
from gee import GEE
import cv2
import base64
import io
import rasterio
from rasterio.transform import xy

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

api_desc:str = "This API for Company Track Capstone in Solafune.inc from C242-FS01 Team."

model = Model("models/seg_v6_3class_12feature.h5")
model.load_model()

gee = GEE()

async def model_predict(image):
    loop = asyncio.get_event_loop()
    with io.BytesIO(image) as image_file:
        rgb_image, mask_image = await loop.run_in_executor(None, model.predict, image_file)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    mask_image = mask_image * 255
    rbg_image_base64 = base64.b64encode(cv2.imencode('.png', rgb_image)[1]).decode()
    mask_image_base64 = base64.b64encode(cv2.imencode('.png', mask_image)[1]).decode()
    return rbg_image_base64, mask_image_base64

@app.post("/api/image_predict")
async def image_predict(file: UploadFile = File(...)):
    try:
        image_content = await file.read()

        if file.content_type != "image/tiff":
            return JSONResponse(status_code=400, content={"message": "Only TIFF Image are supported"})
        
        with rasterio.MemoryFile(image_content) as memfile:
            with memfile.open() as dataset:
                if dataset.count != 12:
                    return JSONResponse(status_code=400, content={"message": "TIFF Image does not have 12 bands"})
                transform = dataset.transform
        
        row, col = 0, 0
        long, lat = xy(transform, row, col)
        image, mask = await asyncio.create_task(model_predict(image_content))
        
        return {
            "message": "Field mask succesfully predicted!",
            "content": {
                "image": image,
                "mask": mask,
                "long": long,
                "lat": lat,
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/api/gee_predict")
async def gee_predict(long:float, lat:float, radius:int=3000):
    try:
        if (long <= -180 or long >= 180):
            return JSONResponse(status_code=400, content={
                "message":  "Longtitude must between -180 to 180"
            })
        if (lat <= -90 or lat >= 90):
            return JSONResponse(status_code=400, content={
                "message":  "Latitude must between -90 to 90"
            })
        
        image_file = gee.get_tiff_content(long=long, lat=lat, radius=radius, save=False)
        
        image, mask = await asyncio.create_task(model_predict(image_file))
        
        return {
            "message": "Field mask succesfully predicted!",
            "content": {
                "image": image,
                "mask": mask,
                "long": long,
                "lat": lat,
                "radius": radius
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/")
async def root():
    return {
        "message": api_desc,
        "model_status": "Model loaded sucessfully!" if model.status == "loaded" else "Model not loaded"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True)
