# import easyocr
# from image_utils import preprocess_image
# from thai_extractor import extract_thai_id_info

# class OCREngine:
#     def __init__(self):
#         # Initialize with Thai and English
#         self.reader = easyocr.Reader(['th', 'en'])

#     def process_id_card(self, image_path):
#         # 1. Pre-process (Crop/Rotate/Perspective)
#         processed_img = preprocess_image(image_path)
        
#         # 2. OCR Read with detailed output (boxes, text, confidence)
#         detailed_results = self.reader.readtext(processed_img)
        
#         # detailed_results format: [([[x,y],[x,y],[x,y],[x,y]], "text", confidence), ...]
        
#         # 3. Extract Data using detailed spatial info
#         info = extract_thai_id_info(detailed_results)
        
#         return info

import os
import cv2
import yaml 
import numpy as np
import easyocr
import base64
from pathlib import Path
from collections import namedtuple
from core.utils import Language, Provider, Card, remove_dot_noise


class PersonalCard:
    def __init__(self,lang=Language.MIX , provider = Provider.EASYOCR):
        self.lang = lang
        # กำหนด OCR Engine ที่จะใช้งาน 
        self.provider = provider
        # หา Path ของโฟลเดอร์หลักโปรเจกต์ (ย้อนกลับไป 2 ระดับจากไฟล์นี้) เพื่อใช้สำหรับอ้างอิงอ่านไฟล์อื่นๆ
        self.root_path = Path(__file__).parent.parent
        # โหลดโมเดล EasyOCR สำหรับอ่านภาษาไทยและอังกฤษมารอไว้ พร้อมเปิดใช้ GPU (เพื่อให้ประมวลผลได้เร็ว)
        self.reader = easyocr.Reader(['th','en'],gpu=True)
        # สร้าง Matcher จาก OpenCV สำหรับจับคู่จุดสนใจบนภาพด้วยอัลกอริทึม FLANN (ใช้เพื่อจัดตำแหน่งบัตรเอียงให้ตรง)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, tree = 5),dict())
        # สร้างตัวดึงจุดเด่นของภาพ (Feature Extractor) ด้วยอัลกอริทึม SIFT โดยตั้งขีดจำกัดดึงสูงสุด 25,000 จุด
        self.sift = cv2.SIFT_create(25000)
        # เรียกใช้งานฟังก์ชัน (Private Method) เพื่อโหลดทรัพยากรอื่นๆ เพิ่มเติม เช่น โหลด config.yaml หรือภาพ Template 
        self.__load_resources()

    def __load_resources(self):
        # load template ของบัตรประชาชนด้านหน้า
        template_path = os.path.join(self.root_path,'core/dataset/indentity_card/personal-card-template.jpg')
        self.template_img = cv2.imread(template_path)
        if self.template_img is None:
            raise ValueError(f"Template image not found at {template_path}")
        
        # แปลงสีภาพจาก BGR (ค่าเริ่มต้นของ OpenCV) เป็น RGB
        self.template_img = cv2.cvtColor(self.template_img,cv2.COLOR_BGR2RGB)
        # เก็บค่าความสูง (h) และความกว้าง (w) ของภาพต้นแบบไว้ใช้ตอนจัดรูปทรง
        self.h , self.w = self.template_img.shape[:2]

        # คำนวณหาจุดเด่น (Keypoints) และรายละเอียดของจุด (Descriptors) ของภาพต้นแบบรอไว้เลย จะได้ไม่ต้องทำใหม่ทุกครั้ง
        self.template_kp , self.temp_des = self.sift.detectAndCompute(self.template_img,None)

        # โหลดไฟล์ config.yaml ที่เก็บข้อมูลพิกัด (ROI) ว่าข้อมูลแต่ละอย่างของบัตรอยู่ตำแหน่งไหน
        config_path = os.path.join(self.root_path,'core/dataset/indentity_card/config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def extract_front_info(self,image_bytes):
        # แปลงข้อมูลไฟล์ภาพที่รับมาแบบ Bytes ให้กลายเป็นเมทริกซ์ภาพของ OpenCV
        nparr = np.frombuffer(image_bytes,np.uint8)
        img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # 2. SIFT Matching: หาจุดเด่นของภาพที่อัปโหลดมา และนำไปจับคู่กับจุดเด่นของภาพต้นแบบ
        kp,des = self.sift.detectAndCompute(img,None)
        matches = self.flann.knnMatch(des,self.temp_des, k=2)
        # คัดกรองเฉพาะจุดที่จับคู่กันได้แม่นยำจริงๆ (Lowe's ratio test)
        good = [m for m , n in matches if m.distance < 0.7 * n.distance]
        
        if  len(good) > 30 :
            # ดึงพิกัดของจุดที่จับคู่ได้
            src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([self.template_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            # คำนวณสมการ Homography เพื่อใช้ดัดรูปทรงภาพ
            M , _  = cv2.findHomography(src_pts,dst_pts , cv2.RANSAC , 5.0)
            # ดัดภาพให้ตรงและมีขนาดเท่ากับภาพต้นแบบเป๊ะๆ
            aligned_img = cv2.warpPerspective(img,M, (self.w, self.h))
        else:
            aligned_img = img # ถ้าจับคู่จุดได้น้อยเกินไป แสดงว่าหาบัตรไม่เจอ ให้ใช้ภาพเดิมไปก่อน
        
        # วนลูปตัดภาพตามพิกัด ROI จาก config.yaml และให้ EasyOCR อ่านข้อความ
        results = {}
        for  roi in self.config['roi_extract']["front"]:
            x1,y1,x2,y2 = roi["point"]
            # ตัดภาพ (Crop) ตามพิกัด [y1:y2, x1:x2]
            crop = aligned_img[y1:y2,x1:x2]

            # แปลงภาพที่ตัดมาเป็นขาวดำ (Grayscale) เพื่อให้ EasyOCR อ่านตัวหนังสือได้แม่นยำขึ้น
            crop_gray = cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
            text = self.reader.readtext(crop_gray, detail=0, paragraph=False) 
            # นำภาพไปลบจุดรบกวนก่อนส่งให้ OCR
            crop_clean = remove_dot_noise(crop_gray)
            text = self.reader.readtext(crop_clean, detail=0, paragraph=False) 
            # text = ['นาย', 'สมชาย', 'ใจดี']  
            extracted_text =  " ".join(text).strip() #"นาย สมชาย ใจดี"

            
            if roi['name'] == "FullNameThai" and extracted_text:
                # เเยกคำด้วยช่องว่าง
                parts = extracted_text.split()
                if len(parts) >= 2:
                    results["PrefixTH"] = parts[0]
                    results["NameTH"] = parts[1] if len(parts) > 2 else parts[-1]
                    results["LastNameTH"] = parts[-1]
                else:
                    results["FullNameTH_Raw"] = extracted_text  # เผื่อเเยกไฟล์ไม่ได้
            else:
                results[roi["name"]] = extracted_text
 
        return results 