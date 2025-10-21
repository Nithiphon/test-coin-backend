from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
import logging
from datetime import datetime

# ลด memory usage
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy loading สำหรับโมเดล
model = None

def load_model():
    """โหลดโมเดลเมื่อจำเป็นเท่านั้น"""
    global model
    if model is None:
        logger.info("🔄 Loading YOLO model...")
        from ultralytics import YOLO
        try:
            model = YOLO('model-coin.pt')
            logger.info("✅ Model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
    return model

@app.route('/')
def home():
    return jsonify({
        'status': 'active',
        'message': 'Coin Detection API is running!',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health_check():
    try:
        model = load_model()
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/detect', methods=['POST'])
def detect_coins():
    start_time = datetime.now()
    logger.info("📥 Received detection request")
    
    try:
        # ตรวจสอบไฟล์
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        # ตรวจสอบขนาดไฟล์ (ไม่เกิน 2MB)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 2 * 1024 * 1024:
            return jsonify({
                'success': False, 
                'error': 'File too large (max 2MB)'
            }), 400

        logger.info(f"📄 Processing file: {file.filename} ({file_size} bytes)")

        # อ่านและ resize รูปภาพเพื่อลด memory
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'error': 'Invalid image file'}), 400

        # Resize image ถ้าใหญ่เกิน (max 1000px)
        height, width = img.shape[:2]
        if max(height, width) > 1000:
            scale = 1000 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            logger.info(f"📏 Resized image: {width}x{height} -> {new_width}x{new_height}")

        # โหลดโมเดลและประมวลผล
        model = load_model()
        
        logger.info("🤖 Running YOLO detection...")
        results = model(img, conf=0.4, verbose=False)  # ลด confidence เพื่อเพิ่ม speed
        
        detections = results[0].boxes
        coin_count = len(detections)
        
        # นับเหรียญ
        coin_types = {'1baht': 0, '5baht': 0, '10baht': 0}
        total_value = 0
        
        for box in detections:
            cls = int(box.cls[0])
            label = model.names[cls]
            
            if label == "1baht":
                coin_types['1baht'] += 1
                total_value += 1
            elif label == "5baht":
                coin_types['5baht'] += 1
                total_value += 5
            elif label == "10baht":
                coin_types['10baht'] += 1
                total_value += 10

        logger.info(f"✅ Found {coin_count} coins, total value: {total_value} THB")

        # วาด bounding boxes (เฉพาะถ้ามีเหรียญ)
        if coin_count > 0:
            img_with_boxes = results[0].plot()
            _, buffer = cv2.imencode('.jpg', img_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            # ถ้าไม่พบเหรียญ ส่งรูปเดิมกลับ
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_base64 = base64.b64encode(buffer).decode('utf-8')

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"⏱️ Processing completed in {processing_time:.2f}s")

        return jsonify({
            'success': True,
            'total_count': coin_count,
            'total_value': total_value,
            'coin_details': coin_types,
            'image_with_boxes': img_base64,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Processing error: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting server on port {port}")
    app.run(debug=False, port=port, host='0.0.0.0')