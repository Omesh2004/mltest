from flask import Flask, jsonify
import os
import pytesseract
from kyc import kyc_verification_endpoint  # Your KYC logic module

app = Flask(__name__)

# --- Hardcoded file paths (in same folder as this file) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AADHAR_PATH = os.path.join(BASE_DIR, "Omesh_Aadhar.jpeg")
PAN_PATH = os.path.join(BASE_DIR, "Omesh_Pan.jpeg")

# --- Try locating Tesseract OCR automatically ---
POSSIBLE_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]

tesseract_found = False
for path in POSSIBLE_TESSERACT_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        tesseract_found = True
        break


@app.route('/verify-kyc', methods=['GET'])
def verify_kyc_route():
    """
    Verify KYC using hardcoded Aadhaar and PAN image paths.
    Provides detailed JSON feedback.
    """

    # 1️⃣ Check if Tesseract OCR is installed
    if not tesseract_found:
        return jsonify({
            "success": False,
            "error": "Tesseract OCR not found on this system.",
            "fix": "Please install from https://github.com/UB-Mannheim/tesseract/wiki",
            "hint": "Ensure it's in PATH or installed at 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'"
        }), 500

    # 2️⃣ Check if Aadhaar and PAN images exist
    missing_files = []
    if not os.path.exists(AADHAR_PATH):
        missing_files.append("Omesh_Aadhar.jpeg not found")
    if not os.path.exists(PAN_PATH):
        missing_files.append("Omesh_Pan.jpeg not found")

    if missing_files:
        return jsonify({
            "success": False,
            "error": "Missing required files",
            "missing": missing_files,
            "aadhar_path": AADHAR_PATH,
            "pan_path": PAN_PATH
        }), 404

    # 3️⃣ If everything is fine — run verification
    try:
        result = kyc_verification_endpoint(AADHAR_PATH, PAN_PATH)
        return jsonify({
            "success": True,
            "message": "KYC verification completed successfully",
            "tesseract_found": True,
            "aadhar_image_found": os.path.exists(AADHAR_PATH),
            "pan_image_found": os.path.exists(PAN_PATH),
            "data": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "tesseract_found": True,
            "aadhar_image_found": os.path.exists(AADHAR_PATH),
            "pan_image_found": os.path.exists(PAN_PATH)
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
