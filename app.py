from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
import logging
from datetime import datetime
import gc

# ลด memory usage
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

app = Flask(__name__)
CORS(app)

# เพิ่ม limit สำหรับไฟล์ใหญ่ (50MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

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

def optimize_image(img, max_size=2000, quality=85):
    """
    ปรับขนาดและคุณภาพรูปภาพเพื่อลด memory usage
    """
    height, width = img.shape[:2]
    
    # ถ้ารูปใหญ่เกิน ให้ resize
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        logger.info(f"📏 Resizing image: {width}x{height} -> {new_width}x{new_height}")
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return img

def process_large_image(img_bytes, max_dimension=2000):
    """
    ประมวลผลรูปภาพใหญ่โดยใช้ memory น้อยลง
    """
    try:
        # Decode image
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Cannot decode image")
        
        # Optimize image size
        img_optimized = optimize_image(img, max_dimension)
        
        # Clear memory
        del img
        gc.collect()
        
        return img_optimized
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise

@app.route('/')
def home():
    return jsonify({
        'status': 'active',
        'message': 'Coin Detection API is running!',
        'timestamp': datetime.now().isoformat(),
        'max_file_size': '50MB',
        'features': ['auto-resize', 'memory-optimized']
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

        # ตรวจสอบขนาดไฟล์ (เพิ่มเป็น 50MB)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'success': False, 
                'error': f'File too large (max {MAX_FILE_SIZE//1024//1024}MB)'
            }), 400

        logger.info(f"📄 Processing file: {file.filename} ({file_size//1024} KB)")

        # อ่านไฟล์
        img_bytes = file.read()
        
        # ประมวลผลรูปภาพ (รองรับไฟล์ใหญ่)
        img = process_large_image(img_bytes, max_dimension=2000)

        # โหลดโมเดลและประมวลผล
        model = load_model()
        
        logger.info("🤖 Running YOLO detection...")
        results = model(img, conf=0.4, verbose=False)
        
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

        # วาด bounding boxes (optimized สำหรับรูปใหญ่)
        if coin_count > 0:
            img_with_boxes = results[0].plot()
        else:
            img_with_boxes = img

        # Encode image ด้วย quality ที่เหมาะสม
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        _, buffer = cv2.imencode('.jpg', img_with_boxes, encode_params)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Clear memory
        del img, results, detections
        if 'img_with_boxes' in locals():
            del img_with_boxes
        gc.collect()

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"⏱️ Processing completed in {processing_time:.2f}s")

        return jsonify({
            'success': True,
            'total_count': coin_count,
            'total_value': total_value,
            'coin_details': coin_types,
            'image_with_boxes': img_base64,
            'processing_time': processing_time,
            'file_size_original': file_size,
            'timestamp': datetime.now().isoformat()
        })
        
    except MemoryError:
        logger.error("💥 Memory error - file too large")
        return jsonify({
            'success': False,
            'error': 'File too large - out of memory. Please try with a smaller image.'
        }), 500
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Processing error: {str(e)}'
        }), 500

# เพิ่ม endpoint สำหรับตรวจสอบข้อมูล
@app.route('/info')
def info():
    return jsonify({
        'service': 'Coin Detection API',
        'max_file_size': '50MB',
        'supported_formats': ['JPEG', 'JPG', 'PNG'],
        'features': [
            'Auto image resizing',
            'Memory optimized',
            'Large file support',
            'CORS enabled'
        ],
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting server on port {port}")
    logger.info(f"💾 Max file size: 50MB")
    app.run(debug=False, port=port, host='0.0.0.0')