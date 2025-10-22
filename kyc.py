"""
AI-based KYC Checker with OCR
Supports Aadhar Card and PAN Card verification with fraud detection
"""

import re
import cv2
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image
import json
from typing import Dict, List, Tuple, Optional

# Optional: Set tesseract path if not in system PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class KYCChecker:
    """
    Comprehensive KYC verification system with OCR and fraud detection
    """
    
    def __init__(self):
        self.fraud_weights = {
            'name_mismatch': 30,
            'dob_mismatch': 40,
            'format_invalid': 50,
            'low_confidence': 20,
            'duplicate_document': 60,
            'age_invalid': 70
        }
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Dilation and erosion to remove noise
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(thresh, kernel, iterations=1)
        processed = cv2.erode(processed, kernel, iterations=1)
        
        return processed
    
    def extract_text_from_image(self, image_path: str) -> Tuple[str, float]:
        """
        Extract text from image using OCR with confidence score
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Convert to PIL Image for pytesseract
            pil_img = Image.fromarray(processed_img)
            
            # Perform OCR with confidence data
            ocr_data = pytesseract.image_to_data(
                pil_img, 
                output_type=pytesseract.Output.DICT,
                lang='eng'
            )
            
            # Calculate average confidence
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Extract text
            text = pytesseract.image_to_string(pil_img, lang='eng')
            
            return text, avg_confidence / 100.0
            
        except Exception as e:
            raise Exception(f"OCR extraction failed: {str(e)}")
    
    def parse_aadhar_card(self, image_path: str) -> Dict:
        """
        Extract information from Aadhar card
        """
        text, confidence = self.extract_text_from_image(image_path)
        
        # Initialize data dictionary
        data = {
            'document_type': 'AADHAR',
            'raw_text': text,
            'confidence': confidence,
            'name': None,
            'aadhar_number': None,
            'dob': None,
            'gender': None,
            'address': None
        }
        
        lines = text.split('\n')
        text_clean = ' '.join(lines)
        
        # Extract Aadhar number (12 digits, may have spaces)
        aadhar_patterns = [
            r'\b\d{4}\s?\d{4}\s?\d{4}\b',
            r'\b\d{12}\b'
        ]
        for pattern in aadhar_patterns:
            match = re.search(pattern, text_clean)
            if match:
                aadhar_num = re.sub(r'\s+', '', match.group())
                data['aadhar_number'] = f"{aadhar_num[:4]}-{aadhar_num[4:8]}-{aadhar_num[8:]}"
                break
        
        # Extract DOB (various formats)
        dob_patterns = [
            r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b',
            r'\b(\d{2}\s+\d{2}\s+\d{4})\b',
            r'DOB[:\s]+(\d{2}[/-]\d{2}[/-]\d{4})',
            r'Birth[:\s]+(\d{2}[/-]\d{2}[/-]\d{4})'
        ]
        for pattern in dob_patterns:
            match = re.search(pattern, text_clean, re.IGNORECASE)
            if match:
                dob_str = match.group(1)
                data['dob'] = re.sub(r'[\s]+', '/', dob_str.replace('-', '/'))
                break
        
        # Extract Gender
        if re.search(r'\b(MALE|Male)\b', text_clean):
            data['gender'] = 'Male'
        elif re.search(r'\b(FEMALE|Female)\b', text_clean):
            data['gender'] = 'Female'
        
        # Extract name (usually after "Name" keyword or in first few lines)
        name_patterns = [
            r'Name[:\s]+([A-Z][A-Za-z\s]+?)(?:\n|DOB|Male|Female|\d)',
            r'^([A-Z][A-Z\s]+)$'
        ]
        for line in lines[:5]:  # Check first 5 lines
            if len(line.strip()) > 5 and line.strip().isupper():
                # Likely a name in capital letters
                potential_name = line.strip()
                if not re.search(r'\d', potential_name):  # No digits
                    data['name'] = potential_name
                    break
        
        # Extract address (lines after common keywords)
        address_lines = []
        capture_address = False
        for line in lines:
            if re.search(r'address|s/o|d/o|c/o', line, re.IGNORECASE):
                capture_address = True
            if capture_address and len(line.strip()) > 10:
                address_lines.append(line.strip())
                if len(address_lines) >= 3:
                    break
        
        if address_lines:
            data['address'] = ' '.join(address_lines[:3])
        
        return data
    
    def parse_pan_card(self, image_path: str) -> Dict:
        """
        Extract information from PAN card
        """
        text, confidence = self.extract_text_from_image(image_path)
        
        data = {
            'document_type': 'PAN',
            'raw_text': text,
            'confidence': confidence,
            'name': None,
            'pan_number': None,
            'dob': None,
            'father_name': None
        }
        
        lines = text.split('\n')
        text_clean = ' '.join(lines)
        
        # Extract PAN number (5 letters, 4 digits, 1 letter)
        pan_match = re.search(r'\b([A-Z]{5}\d{4}[A-Z])\b', text_clean)
        if pan_match:
            data['pan_number'] = pan_match.group(1)
        
        # Extract DOB
        dob_patterns = [
            r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b',
            r'(\d{2}/\d{2}/\d{4})'
        ]
        for pattern in dob_patterns:
            match = re.search(pattern, text_clean)
            if match:
                data['dob'] = match.group(1).replace('-', '/')
                break
        
        # Extract Name (usually in capital letters)
        for line in lines:
            if len(line.strip()) > 5 and line.strip().isupper():
                if not re.search(r'\d', line) and 'INCOME' not in line and 'DEPARTMENT' not in line:
                    if data['name'] is None:
                        data['name'] = line.strip()
                    elif data['father_name'] is None:
                        data['father_name'] = line.strip()
        
        # Look for father's name specifically
        father_match = re.search(r'Father[\'s]*\s+Name[:\s]+([A-Z\s]+)', text_clean, re.IGNORECASE)
        if father_match:
            data['father_name'] = father_match.group(1).strip()
        
        return data
    
    def calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two names using fuzzy matching
        """
        if not name1 or not name2:
            return 0.0
        
        # Normalize names
        n1 = re.sub(r'[^a-zA-Z\s]', '', name1).lower().strip()
        n2 = re.sub(r'[^a-zA-Z\s]', '', name2).lower().strip()
        
        # Calculate similarity
        return SequenceMatcher(None, n1, n2).ratio()
    
    def validate_aadhar_format(self, aadhar_number: str) -> bool:
        """
        Validate Aadhar number format
        """
        if not aadhar_number:
            return False
        pattern = r'^\d{4}-\d{4}-\d{4}$'
        return bool(re.match(pattern, aadhar_number))
    
    def validate_pan_format(self, pan_number: str) -> bool:
        """
        Validate PAN number format
        """
        if not pan_number:
            return False
        pattern = r'^[A-Z]{5}\d{4}[A-Z]$'
        return bool(re.match(pattern, pan_number))
    
    def calculate_age(self, dob_string: str) -> Optional[int]:
        """
        Calculate age from DOB string
        """
        if not dob_string:
            return None
        
        try:
            # Try different date formats
            for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d']:
                try:
                    dob = datetime.strptime(dob_string, fmt)
                    today = datetime.now()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    return age
                except ValueError:
                    continue
            return None
        except Exception:
            return None
    
    def normalize_dob(self, dob: str) -> str:
        """
        Normalize DOB to DD/MM/YYYY format
        """
        if not dob:
            return ""
        
        # Try to parse and reformat
        for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d %m %Y']:
            try:
                date_obj = datetime.strptime(dob.strip(), fmt)
                return date_obj.strftime('%d/%m/%Y')
            except ValueError:
                continue
        
        return dob
    
    def perform_fraud_detection(self, aadhar_data: Dict, pan_data: Dict) -> Dict:
        """
        Perform comprehensive fraud detection
        """
        checks = []
        fraud_score = 0
        warnings = []
        
        # 1. Name Matching
        name_similarity = self.calculate_name_similarity(
            aadhar_data.get('name', ''),
            pan_data.get('name', '')
        )
        name_passed = name_similarity > 0.75
        checks.append({
            'name': 'Name Verification',
            'passed': name_passed,
            'score': round(name_similarity, 3),
            'details': f"{round(name_similarity * 100, 1)}% match"
        })
        if not name_passed:
            fraud_score += self.fraud_weights['name_mismatch']
            warnings.append(f"Name mismatch detected (similarity: {round(name_similarity * 100, 1)}%)")
        
        # 2. DOB Verification
        aadhar_dob = self.normalize_dob(aadhar_data.get('dob', ''))
        pan_dob = self.normalize_dob(pan_data.get('dob', ''))
        dob_match = aadhar_dob == pan_dob if aadhar_dob and pan_dob else False
        checks.append({
            'name': 'Date of Birth',
            'passed': dob_match,
            'details': f"Aadhar: {aadhar_dob}, PAN: {pan_dob}"
        })
        if not dob_match:
            fraud_score += self.fraud_weights['dob_mismatch']
            warnings.append('Date of birth mismatch between documents')
        
        # 3. Document Format Validation
        aadhar_valid = self.validate_aadhar_format(aadhar_data.get('aadhar_number', ''))
        pan_valid = self.validate_pan_format(pan_data.get('pan_number', ''))
        format_valid = aadhar_valid and pan_valid
        checks.append({
            'name': 'Document Format',
            'passed': format_valid,
            'details': f"Aadhar: {'Valid' if aadhar_valid else 'Invalid'}, PAN: {'Valid' if pan_valid else 'Invalid'}"
        })
        if not format_valid:
            fraud_score += self.fraud_weights['format_invalid']
            warnings.append('Invalid document number format detected')
        
        # 4. OCR Confidence Check
        avg_confidence = (aadhar_data.get('confidence', 0) + pan_data.get('confidence', 0)) / 2
        confidence_passed = avg_confidence > 0.70
        checks.append({
            'name': 'OCR Quality',
            'passed': confidence_passed,
            'score': round(avg_confidence, 3),
            'details': f"{round(avg_confidence * 100, 1)}% confidence"
        })
        if not confidence_passed:
            fraud_score += self.fraud_weights['low_confidence']
            warnings.append('Low document quality - unclear images')
        
        # 5. Age Verification
        age = self.calculate_age(aadhar_dob)
        age_valid = age is not None and 18 <= age <= 100
        checks.append({
            'name': 'Age Verification',
            'passed': age_valid,
            'details': f"Age: {age} years" if age else "Could not determine age"
        })
        if not age_valid:
            fraud_score += self.fraud_weights['age_invalid']
            warnings.append('Age does not meet requirements (must be 18-100)')
        
        # 6. Cross-field Validation
        has_all_fields = all([
            aadhar_data.get('name'),
            aadhar_data.get('aadhar_number'),
            aadhar_data.get('dob'),
            pan_data.get('name'),
            pan_data.get('pan_number'),
            pan_data.get('dob')
        ])
        checks.append({
            'name': 'Data Completeness',
            'passed': has_all_fields,
            'details': 'All required fields extracted' if has_all_fields else 'Missing required fields'
        })
        if not has_all_fields:
            fraud_score += 15
            warnings.append('Some required fields could not be extracted')
        
        # Determine status
        if fraud_score == 0:
            status = 'VERIFIED'
            risk_level = 'Low'
        elif fraud_score < 50:
            status = 'WARNING'
            risk_level = 'Medium'
        else:
            status = 'REJECTED'
            risk_level = 'High'
        
        return {
            'status': status,
            'fraud_score': fraud_score,
            'risk_level': risk_level,
            'checks': checks,
            'warnings': warnings,
            'confidence_score': round(avg_confidence, 3)
        }
    
    def verify_kyc(self, aadhar_image_path: str, pan_image_path: str) -> Dict:
        """
        Main KYC verification function
        
        Args:
            aadhar_image_path: Path to Aadhar card image
            pan_image_path: Path to PAN card image
            
        Returns:
            Dictionary containing verification results
        """
        try:
            # Extract data from documents
            print("Extracting Aadhar card data...")
            aadhar_data = self.parse_aadhar_card(aadhar_image_path)
            
            print("Extracting PAN card data...")
            pan_data = self.parse_pan_card(pan_image_path)
            
            print("Performing fraud detection...")
            verification_result = self.perform_fraud_detection(aadhar_data, pan_data)
            
            # Compile final result
            result = {
                'success': True,
                'aadhar_data': aadhar_data,
                'pan_data': pan_data,
                'verification': verification_result,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Flask integration example
def kyc_verification_endpoint(aadhar_path: str, pan_path: str) -> Dict:
    """
    Function ready for Flask integration
    
    Usage in Flask:
    @app.route('/verify-kyc', methods=['POST'])
    def verify_kyc_route():
        aadhar_file = request.files['aadhar']
        pan_file = request.files['pan']
        
        # Save files temporarily
        aadhar_path = f"/tmp/{aadhar_file.filename}"
        pan_path = f"/tmp/{pan_file.filename}"
        aadhar_file.save(aadhar_path)
        pan_file.save(pan_path)
        
        # Verify KYC
        result = kyc_verification_endpoint(aadhar_path, pan_path)
        
        # Clean up
        os.remove(aadhar_path)
        os.remove(pan_path)
        
        return jsonify(result)
    """
    checker = KYCChecker()
    return checker.verify_kyc(aadhar_path, pan_path)


# Example usage
if __name__ == "__main__":
    # Initialize checker
    checker = KYCChecker()
    
    # Verify documents
    result = checker.verify_kyc(
        aadhar_image_path="path/to/aadhar.jpg",
        pan_image_path="path/to/pan.jpg"
    )
    
    # Print results
    print(json.dumps(result, indent=2))
    
    # Access specific data
    if result['success']:
        print(f"\nVerification Status: {result['verification']['status']}")
        print(f"Risk Level: {result['verification']['risk_level']}")
        print(f"Fraud Score: {result['verification']['fraud_score']}/100")
        print(f"\nExtracted Name (Aadhar): {result['aadhar_data']['name']}")
        print(f"Extracted Name (PAN): {result['pan_data']['name']}")
    else:
        print(f"Error: {result['error']}")