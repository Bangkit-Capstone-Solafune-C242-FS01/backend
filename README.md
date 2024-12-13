# FastAPI API Documentation

This repository contains the implementation of an API designed for the **Company Track Capstone** at Solafune Inc. by the **C242-FS01 Team**. The API is built using FastAPI and includes functionality for image prediction and geographic data retrieval.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
  - [Root Endpoint](#root-endpoint)
  - [Image Prediction Endpoint](#image-prediction-endpoint)
  - [GEE Prediction Endpoint](#gee-prediction-endpoint)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features
- **Model Prediction**: Processes uploaded TIFF images with 12 bands to generate RGB and mask predictions.
- **Geographic Data Retrieval**: Retrieves geospatial data based on longitude, latitude, and radius using Google Earth Engine (GEE).
- **CORS Support**: Configured to allow cross-origin requests for integration with frontend applications.

---

## Requirements
- Python 3.9+
- FastAPI
- Uvicorn
- Rasterio
- OpenCV
- TensorFlow (for model loading and prediction)
- asyncio
- earthengine-api
- python-multipart
- patchify
- numpy
- scipy
- requests

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Bangkit-Capstone-Solafune-C242-FS01/backend.git
   cd backend
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate # For Linux/Mac
   venv\Scripts\activate   # For Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the server**:
   ```bash
   uvicorn app:app --reload
   ```

or Using Dockerfile
  ```bash
  docker build -t c242-fs01/solafune-be
  ```

---

## Usage
After starting the server, you can access the API at `http://127.0.0.1:8000`. Use tools like Postman, curl, or a frontend client to interact with the API.

---

## API Endpoints

### Root Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns the API description and model status.
- **Response**:
  ```json
  {
    "message": "This API for Company Track Capstone in Solafune.inc from C242-FS01 Team.",
    "model_status": "Model loaded successfully!"
  }
  ```

---

### Image Prediction Endpoint
- **URL**: `/api/image_predict`
- **Method**: `POST`
- **Description**: Predicts field masks from a TIFF image with 12 bands.
- **Request Body**:
  - A `multipart/form-data` file upload with a TIFF image.

- **Response**:
  - **Success (200)**:
    ```json
    {
      "message": "Field mask successfully predicted!",
      "content": {
        "image": "<base64-encoded RGB image>",
        "mask": "<base64-encoded mask image>",
        "long": <longitude>,
        "lat": <latitude>
      }
    }
    ```
  - **Error (400)**:
    ```json
    { "message": "Only TIFF Image are supported" }
    ```
    ```json
    { "message": "TIFF Image does not have 12 bands" }
    ```

---

### GEE Prediction Endpoint
- **URL**: `/api/gee_predict`
- **Method**: `POST`
- **Description**: Retrieves a TIFF file using Google Earth Engine and predicts the field mask.
- **Request Body**:
  - `long` (float): Longitude.
  - `lat` (float): Latitude.
  - `radius` (int, optional): Radius in meters (default is 3000).

- **Response**:
  - **Success (200)**:
    ```json
    {
      "message": "Field mask successfully predicted!",
      "content": {
        "image": "<base64-encoded RGB image>",
        "mask": "<base64-encoded mask image>",
        "long": <longitude>,
        "lat": <latitude>,
        "radius": <radius>
      }
    }
    ```
  - **Error (400)**:
    ```json
    { "message": "Longitude must be between -180 and 180" }
    ```
    ```json
    { "message": "Latitude must be between -90 and 90" }
    ```

---

## Project Structure
```
.
├── app.py              # Main FastAPI application
├── model.py            # Model wrapper for loading and prediction
├── gee.py              # Google Earth Engine integration
├── requirements.txt    # Python dependencies
├── README.md           # Documentation file
└── models/
    └── seg_v6_3class_12feature.h5  # Pre-trained segmentation model
```

---

## License
This project is licensed under the [MIT License](LICENSE).

