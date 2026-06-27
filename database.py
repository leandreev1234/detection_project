import sqlite3
from datetime import datetime

DB_PATH = 'history.db'

def get_db_connection():
    """Подключение к базе данных"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Инициализация базы данных"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            laptop_count INTEGER NOT NULL,
            result_image TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def save_request(image_name, laptop_count, result_image):
    """Сохранение запроса в историю"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute('''
        INSERT INTO requests (image_name, timestamp, laptop_count, result_image)
        VALUES (?, ?, ?, ?)
    ''', (image_name, timestamp, laptop_count, result_image))
    
    conn.commit()
    conn.close()

def get_all_requests():
    """Получение всех записей из истории"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM requests ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    
    conn.close()
    return [dict(row) for row in rows]