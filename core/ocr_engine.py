import easyocr
from image_utils import preprocess_image
from thai_extractor import extract_thai_id_info

class OCREngine:
    def __init__(self):
        # Initialize with Thai and English
        self.reader = easyocr.Reader(['th', 'en'])

    def process_id_card(self, image_path):
        # 1. Pre-process (Crop/Rotate/Perspective)
        processed_img = preprocess_image(image_path)
        
        # 2. OCR Read with detailed output (boxes, text, confidence)
        detailed_results = self.reader.readtext(processed_img)
        
        # detailed_results format: [([[x,y],[x,y],[x,y],[x,y]], "text", confidence), ...]
        
        # 3. Extract Data using detailed spatial info
        info = extract_thai_id_info(detailed_results)
        
        return info
