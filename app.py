from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from database import init_db, save_request, get_all_requests
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Настройки
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Создание папки для статики
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Загрузка модели YOLOv8...")

# Используем более точную модель (можно заменить на yolov8x.pt)
model = YOLO('yolov8m.pt')
print("Модель успешно загружена!")

# Инициализация базы данных
init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    history = get_all_requests()
    return render_template('index.html', history=history)

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Неподдерживаемый формат файла'}), 400
    
    try:
        # Чтение изображения через OpenCV
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Не удалось загрузить изображение'}), 400
        
        # Увеличиваем размер для лучшего распознавания
        img_resized = cv2.resize(img, (1280, 1280))
        
        # Инференс модели с настройками
        # conf=0.5 — минимальная уверенность 50%, iou=0.5 — NMS для удаления дубликатов
        results = model(img_resized, conf=0.5, iou=0.5)
        
        # Фильтрация только класса "laptop" (класс 63 в COCO)
        laptop_class_id = 63
        filtered_boxes = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id == laptop_class_id:
                        filtered_boxes.append(box)
        
        # Визуализация результатов
        annotated_img = img.copy()
        
        for box in filtered_boxes:
            # Нормализуем координаты обратно к исходному размеру
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Масштабируем координаты обратно к исходному размеру (1280 -> исходный)
            h, w = img.shape[:2]
            scale_x = w / 1280
            scale_y = h / 1280
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            confidence = float(box.conf[0])
            
            # Рисуем зелёную рамку
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Подпись с классом и уверенностью
            label = f"laptop: {confidence:.2f}"
            cv2.putText(annotated_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Сохранение результата
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_filename = f"result_{timestamp}.jpg"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, annotated_img)
        
        laptop_count = len(filtered_boxes)
        
        # Сохранение в историю (SQLite)
        save_request(file.filename, laptop_count, result_filename)
        
        # Возврат статистики в формате JSON
        return jsonify({
            'success': True,
            'count': laptop_count,
            'result_image': result_filename,
            'message': f'Обнаружено ноутбуков: {laptop_count}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)