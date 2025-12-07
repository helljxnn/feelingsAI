from flask import render_template, request, jsonify
from services.emotion_service import EmotionService

emotion_service = EmotionService()

def index():
    """Renderiza la página principal"""
    return render_template('home.html')

def analyze():
    """Endpoint para analizar mensajes"""
    if request.method == 'POST':
        data = request.get_json()
        message = data.get('message', '')
        
        result = emotion_service.analyze_message(message)
        return jsonify(result)
    
    return jsonify({'error': 'Método no permitido'}), 405
