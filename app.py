from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
from ultralytics import YOLO
import os
import base64
import warnings
from datetime import datetime

# ปิด warning ของ PyTorch
warnings.filterwarnings('ignore')

# สร้าง Flask App
app = Flask(__name__)

# CORS settings ที่ครอบคลุมมากขึ้น
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type"]
    }
})

# เพิ่ม header ในการตอบกลับทุกครั้ง
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Max-Age', '86400')
    return response

# โหลดโมเดล YOLO
print("=" * 50)
print("🔄 กำลังโหลดโมเดล YOLO...")

# ⚠️ เปลี่ยนชื่อไฟล์ตรงนี้ให้ตรงกับไฟล์โมเดลของคุณ
MODEL_PATH = 'model-coin.pt'

try:
    # โหลดโมเดลโดยไม่สนใจ warning
    import torch
    
    # ตั้งค่าให้โหลดโมเดลโดยไม่เช็ค weights_only (รองรับ PyTorch ทุกเวอร์ชัน)
    torch_version = torch.__version__.split('+')[0]
    print(f"📦 PyTorch version: {torch_version}")
    
    # โหลดโมเดล YOLO
    model = YOLO(MODEL_PATH)
    
    print("✅ โหลดโมเดลสำเร็จ!")
    print(f"📦 ไฟล์โมเดล: {MODEL_PATH}")
    print(f"📋 Classes: {model.names}")
except Exception as e:
    print(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    print(f"⚠️  กรุณาตรวจสอบว่าไฟล์ {MODEL_PATH} อยู่ในโฟลเดอร์เดียวกับ app.py")
    import traceback
    traceback.print_exc()
    exit()

print("=" * 50)

# Route หน้าแรก
@app.route('/')
def home():
    return jsonify({
        'status': 'active',
        'service': 'Coin Detection API',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'detect': 'POST /detect',
            'health': 'GET /health'
        },
        'message': '🪙 Coin Detection API is running!'
    })

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'model_classes': list(model.names.values()) if model else [],
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

# Route สำหรับ OPTIONS (CORS preflight) - สำคัญ!
@app.route('/detect', methods=['OPTIONS'])
@cross_origin()
def detect_options():
    return jsonify({'status': 'ok'}), 200

# Route สำหรับตรวจจับเหรียญ
@app.route('/detect', methods=['POST'])
@cross_origin()
def detect_coins():
    try:
        print("\n" + "=" * 50)
        print("📥 รับ Request ใหม่")
        print(f"🕒 เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # ตรวจสอบว่ามีไฟล์รูปส่งมาหรือไม่
        if 'image' not in request.files:
            print("❌ ไม่พบไฟล์รูปภาพใน request")
            return jsonify({
                'success': False,
                'error': 'ไม่พบไฟล์รูปภาพ'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            print("❌ ไม่ได้เลือกไฟล์")
            return jsonify({
                'success': False,
                'error': 'ไม่ได้เลือกไฟล์'
            }), 400
        
        print(f"📄 ชื่อไฟล์: {file.filename}")
        print(f"📊 ขนาดไฟล์: {len(file.read())} bytes")
        
        # Reset file pointer
        file.seek(0)
        
        # อ่านรูปภาพ
        print("🔄 กำลังอ่านรูปภาพ...")
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            print("❌ ไม่สามารถอ่านรูปภาพได้")
            return jsonify({
                'success': False,
                'error': 'ไม่สามารถอ่านรูปภาพได้ กรุณาตรวจสอบไฟล์'
            }), 400
        
        print(f"📐 ขนาดรูป: {img.shape}")
        
        # ประมวลผลด้วย YOLO
        print("🤖 กำลังประมวลผลด้วย YOLO...")
        results = model(img, conf=0.5, verbose=False)
        
        # ดึงข้อมูลการตรวจจับ
        detections = results[0].boxes
        coin_count = len(detections)
        
        # นับจำนวนเหรียญแต่ละชนิด
        coin_types = {
            '1baht': 0,
            '5baht': 0,
            '10baht': 0
        }
        
        total_value = 0
        
        print(f"✅ พบเหรียญทั้งหมด: {coin_count} เหรียญ")
        
        if coin_count > 0:
            print("📊 รายละเอียดเหรียญ:")
            
            for i, box in enumerate(detections):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                
                print(f"   เหรียญที่ {i+1}: {label} (ความมั่นใจ {conf:.2%})")
                
                # นับจำนวนและคำนวณมูลค่า
                if label == "1baht":
                    coin_types['1baht'] += 1
                    total_value += 1
                elif label == "5baht":
                    coin_types['5baht'] += 1
                    total_value += 5
                elif label == "10baht":
                    coin_types['10baht'] += 1
                    total_value += 10
        
        print(f"💰 มูลค่ารวม: {total_value} บาท")
        
        # วาดกรอบบนรูปภาพ
        img_with_boxes = results[0].plot()
        
        # แปลงรูปเป็น base64 เพื่อส่งกลับไป
        _, buffer = cv2.imencode('.jpg', img_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print("🎉 ส่งผลลัพธ์กลับไปแล้ว")
        print("=" * 50 + "\n")
        
        # ส่งผลลัพธ์กลับ
        return jsonify({
            'success': True,
            'total_count': coin_count,
            'total_value': total_value,
            'coin_details': {
                '1baht': coin_types['1baht'],
                '5baht': coin_types['5baht'],
                '10baht': coin_types['10baht']
            },
            'image_with_boxes': img_base64,
            'message': f'พบเหรียญทั้งหมด {coin_count} เหรียญ มูลค่ารวม {total_value} บาท',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 50 + "\n")
        return jsonify({
            'success': False,
            'error': f'เกิดข้อผิดพลาด: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

# Test endpoint สำหรับตรวจสอบการทำงาน
@app.route('/test', methods=['GET', 'POST'])
@cross_origin()
def test_endpoint():
    return jsonify({
        'success': True,
        'message': '✅ Test endpoint is working!',
        'timestamp': datetime.now().isoformat(),
        'method': request.method
    })

# เริ่มต้น Flask Server
if __name__ == '__main__':
    # รับ PORT จาก Environment Variable (สำหรับ Deploy)
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "🚀" * 25)
    print("🚀 เริ่มต้น Coin Detection Server")
    print(f"📍 PORT: {port}")
    print("🪙 รองรับเหรียญ: 1, 5, 10 บาท")
    print("🌐 CORS: Enabled for all origins")
    print("📡 Endpoints:")
    print("   GET  /          - หน้าแรก")
    print("   GET  /health    - ตรวจสอบสุขภาพ")
    print("   POST /detect    - ตรวจจับเหรียญ")
    print("   GET  /test      - ทดสอบการทำงาน")
    print("🌐 กด Ctrl+C เพื่อหยุด Server")
    print("🚀" * 25 + "\n")
    
    # debug=False สำหรับ Production
    app.run(debug=False, port=port, host='0.0.0.0')