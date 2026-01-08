# Face Recognition Project

## Overview
This project provides a face recognition system using ONNX models for detection and recognition, with a PostgreSQL database for storing embeddings. It includes scripts for enrolling new faces and running live recognition.

## Folder Structure
- `src/` — Source code (enroll.py, live.py, utils.py)
- `models/` — ONNX models for face detection and recognition
- `db/` — Database scripts and embeddings
- `data/gallery/` — Gallery images (ignored by git)
- `docker/` — Docker setup files

## Setup Instructions

### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd Face_Recognition
```

### 2. Create and Activate a Virtual Environment
```sh
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Setup PostgreSQL Database
- Ensure PostgreSQL is installed and running.
- Use the provided `docker/init.sql` to initialize the database schema if needed.
- Update database connection settings in `db/db.py` as required.

### 5. Prepare Models
- Download the ONNX models from the github and place it in the `models/` directory:
  - `face_detection_yunet_2023mar.onnx`
  - `face_recognition_sface_2021dec.onnx`

### 6. Enroll Faces
Add images to `data/gallery/<person_name>/` and run:
```sh
python -m src.enroll
```

### 7. Run Live Recognition
```sh
python -m src.live
```

## Notes
- The `data/gallery/` folder is ignored by git (see .gitignore).
- For troubleshooting, check the logs and ensure all dependencies are installed.
- For Docker setup, refer to `docker-compose.yaml` and `docker/init.sql`.

## License
Mustakim Bin Mohsin
