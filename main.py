# main.py จะส่งรูปไปให้ ocr_engine.py เพื่อเริ่มทำงาน


from fastapi import FastAPI , File , UploadFile , HTTPException
from fastapi.responses import JSONResponse
from core.ocr_engine import PersonalCard
import uvicorn

#  สร้าง app FastAPI
app = FastAPI(
    title= 'Thai ID Card OCR API',
    description="API เพื่ออ่านข้อมูลจากบัตรประชาชนไทยด้วย easyocr เเละ sift",
    version= "1,0"
)

# load ai engin มารอไว้ก่อนเปิด server ทำให้ api respone เร็วขึ้น
print("loading .......")
try:
    ocr_engine = PersonalCard()
    print("load model sucessfull")
except Exception as e:
    print(f"Error : {e}")

# Endpoint for check server is alive
@app.post("/extract-id")
async def extract_id_card(file: UploadFile = File(...)):
    # ดัก error if file ที่ upload is not image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400 , detail="pls upload an image file (jpg,png)")

    try:
        # read file image ที่ส่งมาเป็น Bytes Data
        image_bytes = await file.read()

        # ส่งรูปไปให้ engine เพื่อตัดเเละอ่านข้อความ
        result = ocr_engine.extract_front_info(image_bytes)

        # check result is NamedTuple or Dictionary
        final_data = result if isinstance(result,dict) else result._asdict()

        # send json to user
        return JSONResponse(content={
            "status": "success",
            "data": final_data
        })
    except Exception as e:
        raise HTTPException(status_code=500  , detail= f"AI processing Error : {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run("main:app" , host="0.0.0.0",port=8000 , reload=True)