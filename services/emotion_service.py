from models.classifier import EmotionClassifier

class EmotionService:
    def __init__(self):
        self.classifier = EmotionClassifier()
        self.classifier.load_model()
    
    def analyze_message(self, message):
        """Analiza un mensaje y retorna la emoción detectada"""
        if not message or len(message.strip()) == 0:
            return {
                'error': 'El mensaje no puede estar vacío'
            }
        
        try:
            result = self.classifier.predict(message)
            return result
        except Exception as e:
            return {
                'error': f'Error al analizar el mensaje: {str(e)}'
            }
