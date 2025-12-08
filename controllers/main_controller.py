from flask import render_template, request, jsonify
from services.emotion_service import EmotionService

# Usar clasificador basado en reglas (use_ml_model=False)
# Cambiar a True cuando el modelo ML esté bien entrenado
emotion_service = EmotionService(use_ml_model=False)

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
