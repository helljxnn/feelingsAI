from models.classifier import EmotionClassifier
from models.rule_based_classifier import RuleBasedClassifier

class EmotionService:
    def __init__(self, use_ml_model=False):
        """
        Inicializa el servicio de emociones
        
        Args:
            use_ml_model: Si es True, usa el modelo ML. Si es False, usa reglas.
        """
        self.use_ml_model = use_ml_model
        
        if use_ml_model:
            self.classifier = EmotionClassifier()
            self.classifier.load_model()
        else:
            self.classifier = RuleBasedClassifier()
    
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
