# Face Recognition Project Dummy

This project provides a face recognition system using OpenCV, MediaPipe, and deep learning models. It includes scripts for enrolling new faces, live recognition, and utility functions.

## Features
- Face detection and recognition using ONNX models
- Live webcam face recognition
- Face enrollment for new users
- Easy-to-extend Python codebase

## Project Structure
```
Face_Recognition/
├── data/                # Gallery of enrolled faces
│   └── gallery/
│       └── [person_name]/[image_name].jpg
├── models/              # Pretrained ONNX models
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── src/                 # Source code
│   ├── enroll.py        # Script to enroll new faces
│   ├── live.py          # Live face recognition script
│   ├── utils.py         # Utility functions
├── requirements.txt     # Python dependencies
└── .venv/               # Python virtual environment (not included in repo)
```

## Requirements
- Python 3.14.x
- See `requirements.txt` for Python packages

## Setup
1. Clone the repository and navigate to the project folder.
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # Or
   source .venv/bin/activate  # On Linux/Mac
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
### Enroll a New Face
Run the enrollment script to add a new face to the gallery:
```
python src/enroll.py
```

### Live Face Recognition
Run the live recognition script to start webcam-based recognition:
```
python src/live.py
```

## Models
Download the required ONNX models and place them in the `models/` directory:
- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

## Data
- Enrolled faces are stored in the `data/gallery/` directory, organized by user name.

## License
This project is for educational and research purposes.

---
Feel free to modify and extend the code for your needs!
