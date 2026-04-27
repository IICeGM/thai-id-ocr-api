""" เก็บค่า constants เพื่อให้เป็นระเบียบและเรียกใช้งานง่าย และ 
การสร้างฟังก์ชันช่วยเหลือ  สำหรับจัดการกับรูปภาพ  """

from enum import Enum
import cv2
import numpy as np

# กำหนดภาษาให้ ocr อ่าน
class Language(Enum):
    THAI = 'tha'
    ENGLISH = 'eng'
    MIX = 'mix'

# ระบุ engine ที่ใช้ทำ ocr 
class Provider(Enum):
    EASYOCR = 'easyocr'
    TESSERACT = 'tesseract'
    DEFAULT = 'default'

# ระบุประเภทของ template card ที่ใช้เเค่ด้านหน้า
class Card(Enum):
    FRONT_TEMPLATE = "front"

def remove_dot_noise(img):
    return cv2.medianBlur(img,3)