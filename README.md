 Smart KYC Checker 

Short summary
- A minimal AI‑powered KYC verification demo using OCR + heuristics.
- Components:
  - kyc.py — main KYC logic (OCR, parsing, fraud checks, scoring).
  - main.py — Flask server endpoint that calls the verifier using hardcoded image files: `Omesh_Aadhar.jpeg` and `Omesh_Pan.jpeg` in the same folder.
  - app.py & DockerFile — a tiny Pathway "Hello World" example (not the KYC service).
- This README explains the architecture, logic, how to run, and troubleshooting.

Prerequisites
- Python 3.8+ (tested with 3.11)
- Tesseract OCR installed on the host:
  - Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
  - Ensure path like `C:\Program Files\Tesseract-OCR\tesseract.exe` is available or in PATH.
- Python packages (install via pip):
  - opencv-python, pillow, pytesseract, flask, numpy

Install Python packages (PowerShell)
```powershell
python -m pip install --upgrade pip
pip install opencv-python pillow pytesseract flask numpy
```

Files and purpose
- c:\Users\DELL\Desktop\react\path\kyc.py
  - KYCChecker class: orchestrates preprocessing, OCR, parsing and fraud detection.
  - `parse_aadhar_card`, `parse_pan_card`: field extraction heuristics (regex + line heuristics).
  - `extract_text_from_image`: uses pytesseract (with `image_to_data`) and returns avg OCR confidence.
  - `perform_fraud_detection`: compares names, DOBs, validates ID formats, computes fraud score & risk.
  - `kyc_verification_endpoint(aadhar_path, pan_path)`: convenience wrapper for Flask.
- c:\Users\DELL\Desktop\react\path\main.py
  - Flask app with single endpoint `/verify-kyc` (GET).
  - Uses hardcoded image paths:
    - `Omesh_Aadhar.jpeg`
    - `Omesh_Pan.jpeg`
  - Checks for Tesseract, files existence, then calls `kyc_verification_endpoint`.
- c:\Users\DELL\Desktop\react\path\app.py
  - Pathway "Hello, World!" sample (not used for KYC).
- c:\Users\DELL\Desktop\react\path\DockerFile
  - Current Dockerfile runs `app.py` (Pathway). It does not containerize the Flask KYC server or include tesseract.

How the KYC verification works (logic overview)
1. Preprocessing
   - Read image via OpenCV, convert to grayscale, denoise, threshold and small morphology to clean text regions.
2. OCR
   - Run pytesseract on preprocessed image.
   - Use `image_to_data` to compute an average OCR confidence score (returned as 0..1).
3. Field extraction (heuristics & regex)
   - Aadhar: look for 12 digits or groups `\d{4}\s?\d{4}\s?\d{4}`, DOB patterns, gender tokens, name lines in uppercase.
   - PAN: look for PAN regex `[A-Z]{5}\d{4}[A-Z]`, DOB patterns, uppercase name lines.
   - Address and father name: best-effort capture using keywords.
4. Validation & fraud checks
   - Name similarity: fuzzy/SequenceMatcher ratio; threshold ~0.75 passes.
   - DOB equality after normalization.
   - Format validation: aadhar `^\d{4}-\d{4}-\d{4}$` (the code formats found number to this shape); PAN `[A-Z]{5}\d{4}[A-Z]`.
   - OCR quality: average confidence threshold ~0.70.
   - Age check: derived from DOB, must be 18..100.
   - Aggregates these into a `fraud_score` using weights and returns `status` (`VERIFIED`, `WARNING`, `REJECTED`) and `risk_level`.

Running locally (dev)
1. Put your test images in the same folder:
   - `c:\Users\DELL\Desktop\react\path\Omesh_Aadhar.jpeg`
   - `c:\Users\DELL\Desktop\react\path\Omesh_Pan.jpeg`
2. Start server (PowerShell):
```powershell
cd c:\Users\DELL\Desktop\react\path
python main.py
```
3. Visit in browser or Postman:
```
GET http://127.0.0.1:5000/verify-kyc
```
4. Response: JSON containing `aadhar_data`, `pan_data`, `verification` (checks, fraud_score, status), and helper booleans.

Sample response structure
```json
{
  "success": true,
  "message": "KYC verification completed successfully",
  "data": {
    "aadhar_data": { "name": "...", "aadhar_number": "...", "dob": "...", "confidence": 0.85, ... },
    "pan_data": { "name": "...", "pan_number": "...", "dob": "...", "confidence": 0.82, ... },
    "verification": {
      "status": "VERIFIED",
      "fraud_score": 10,
      "risk_level": "Low",
      "checks": [ { "name": "Name Verification", "passed": true, ... }, ... ],
      "warnings": []
    }
  }
}
```

Important notes & troubleshooting
- Tesseract not installed: `main.py` will return a helpful JSON error. Install Tesseract and ensure the path is correct.
- OCR quality: photos with blur, low contrast, or rotated scans produce low confidence -> results may be incomplete or wrong. Try better images or adjust preprocessing.
- Hardcoded images: `main.py` currently ignores uploaded files and uses the two hardcoded filenames. To accept uploads, change route to accept POST and save incoming files then call `kyc_verification_endpoint`.
- Docker: current DockerFile runs `app.py` (Pathway Hello World). To containerize Flask/KYC you must:
  - Use a Python base image (or pathway if you add dependencies) and install Tesseract into the image (non-trivial on Windows host).
  - Copy `main.py`, `kyc.py`, and images into the image and expose port 5000.
  - Example quick Dockerfile snippet (not included here) must install `tesseract-ocr` in the image.

Suggested improvements / next steps
- Replace heuristic parsing with a small ML model or use table/field-specific OCR (layout parsing).
- Use `easyocr` or Google Vision API for better OCR on photos.
- Add unit tests for parsing functions (regex edge cases).
- Allow uploads via POST and secure temporary storage.
- Improve name matching using Natural Language name normalization and handle initials / ordering.
- Add logging instead of prints and return structured error codes.

Security & privacy warning
- This example is for learning/demo purposes only. Do NOT use as-is in production for real PII. Apply encryption, secure storage, access control, rate-limiting, and legal/privacy compliance before real use.

If you want, I can:
- Provide a Dockerfile that containers the Flask service with Tesseract (Linux-based image).
- Change the Flask route to accept file uploads (POST) and temporary save/cleanup.
- Create a requirements.txt or a ready-to-run Jupyter notebook demo.

```// filepath: c:\Users\DELL\Desktop\react\path\README.md

# Smart KYC Checker — README

Short summary
- A minimal AI‑powered KYC verification demo using OCR + heuristics.
- Components:
  - kyc.py — main KYC logic (OCR, parsing, fraud checks, scoring).
  - main.py — Flask server endpoint that calls the verifier using hardcoded image files: `Omesh_Aadhar.jpeg` and `Omesh_Pan.jpeg` in the same folder.
  - app.py & DockerFile — a tiny Pathway "Hello World" example (not the KYC service).
- This README explains the architecture, logic, how to run, and troubleshooting.

Prerequisites
- Python 3.8+ (tested with 3.11)
- Tesseract OCR installed on the host:
  - Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
  - Ensure path like `C:\Program Files\Tesseract-OCR\tesseract.exe` is available or in PATH.
- Python packages (install via pip):
  - opencv-python, pillow, pytesseract, flask, numpy

Install Python packages (PowerShell)
```powershell
python -m pip install --upgrade pip
pip install opencv-python pillow pytesseract flask numpy
```

Files and purpose
- c:\Users\DELL\Desktop\react\path\kyc.py
  - KYCChecker class: orchestrates preprocessing, OCR, parsing and fraud detection.
  - `parse_aadhar_card`, `parse_pan_card`: field extraction heuristics (regex + line heuristics).
  - `extract_text_from_image`: uses pytesseract (with `image_to_data`) and returns avg OCR confidence.
  - `perform_fraud_detection`: compares names, DOBs, validates ID formats, computes fraud score & risk.
  - `kyc_verification_endpoint(aadhar_path, pan_path)`: convenience wrapper for Flask.
- c:\Users\DELL\Desktop\react\path\main.py
  - Flask app with single endpoint `/verify-kyc` (GET).
  - Uses hardcoded image paths:
    - `Omesh_Aadhar.jpeg`
    - `Omesh_Pan.jpeg`
  - Checks for Tesseract, files existence, then calls `kyc_verification_endpoint`.
- c:\Users\DELL\Desktop\react\path\app.py
  - Pathway "Hello, World!" sample (not used for KYC).
- c:\Users\DELL\Desktop\react\path\DockerFile
  - Current Dockerfile runs `app.py` (Pathway). It does not containerize the Flask KYC server or include tesseract.

How the KYC verification works (logic overview)
1. Preprocessing
   - Read image via OpenCV, convert to grayscale, denoise, threshold and small morphology to clean text regions.
2. OCR
   - Run pytesseract on preprocessed image.
   - Use `image_to_data` to compute an average OCR confidence score (returned as 0..1).
3. Field extraction (heuristics & regex)
   - Aadhar: look for 12 digits or groups `\d{4}\s?\d{4}\s?\d{4}`, DOB patterns, gender tokens, name lines in uppercase.
   - PAN: look for PAN regex `[A-Z]{5}\d{4}[A-Z]`, DOB patterns, uppercase name lines.
   - Address and father name: best-effort capture using keywords.
4. Validation & fraud checks
   - Name similarity: fuzzy/SequenceMatcher ratio; threshold ~0.75 passes.
   - DOB equality after normalization.
   - Format validation: aadhar `^\d{4}-\d{4}-\d{4}$` (the code formats found number to this shape); PAN `[A-Z]{5}\d{4}[A-Z]`.
   - OCR quality: average confidence threshold ~0.70.
   - Age check: derived from DOB, must be 18..100.
   - Aggregates these into a `fraud_score` using weights and returns `status` (`VERIFIED`, `WARNING`, `REJECTED`) and `risk_level`.

Running locally (dev)
1. Put your test images in the same folder:
   - `c:\Users\DELL\Desktop\react\path\Omesh_Aadhar.jpeg`
   - `c:\Users\DELL\Desktop\react\path\Omesh_Pan.jpeg`
2. Start server (PowerShell):
```powershell
cd c:\Users\DELL\Desktop\react\path
python main.py
```
3. Visit in browser or Postman:
```
GET http://127.0.0.1:5000/verify-kyc
```
4. Response: JSON containing `aadhar_data`, `pan_data`, `verification` (checks, fraud_score, status), and helper booleans.

Sample response structure
```json
{
  "success": true,
  "message": "KYC verification completed successfully",
  "data": {
    "aadhar_data": { "name": "...", "aadhar_number": "...", "dob": "...", "confidence": 0.85, ... },
    "pan_data": { "name": "...", "pan_number": "...", "dob": "...", "confidence": 0.82, ... },
    "verification": {
      "status": "VERIFIED",
      "fraud_score": 10,
      "risk_level": "Low",
      "checks": [ { "name": "Name Verification", "passed": true, ... }, ... ],
      "warnings": []
    }
  }
}
```

Important notes & troubleshooting
- Tesseract not installed: `main.py` will return a helpful JSON error. Install Tesseract and ensure the path is correct.
- OCR quality: photos with blur, low contrast, or rotated scans produce low confidence -> results may be incomplete or wrong. Try better images or adjust preprocessing.
- Hardcoded images: `main.py` currently ignores uploaded files and uses the two hardcoded filenames. To accept uploads, change route to accept POST and save incoming files then call `kyc_verification_endpoint`.
- Docker: current DockerFile runs `app.py` (Pathway Hello World). To containerize Flask/KYC you must:
  - Use a Python base image (or pathway if you add dependencies) and install Tesseract into the image (non-trivial on Windows host).
  - Copy `main.py`, `kyc.py`, and images into the image and expose port 5000.
  - Example quick Dockerfile snippet (not included here) must install `tesseract-ocr` in the image.

Suggested improvements / next steps
- Replace heuristic parsing with a small ML model or use table/field-specific OCR (layout parsing).
- Use `easyocr` or Google Vision API for better OCR on photos.
- Add unit tests for parsing functions (regex edge cases).
- Allow uploads via POST and secure temporary storage.
- Improve name matching using Natural Language name normalization and handle initials / ordering.
- Add logging instead of prints and return structured error codes.

Security & privacy warning
- This example is for learning/demo purposes only. Do NOT use as-is in production for real PII. Apply encryption, secure storage, access control, rate-limiting, and legal/privacy compliance before real use.

If you want, I can:
- Provide a Dockerfile that containers the Flask service with Tesseract (Linux-based image).
- Change the Flask route to accept file uploads (POST) and temporary save/cleanup.
- Create a requirements.txt or a ready-to-run Jupyter notebook demo.
