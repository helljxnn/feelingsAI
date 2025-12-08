import re

class RuleBasedClassifier:
    """Clasificador basado en reglas para emociones"""
    
    def __init__(self):
        # Palabras clave para cada emoción
        self.keywords = {
            'enamoramiento': [
                'amo', 'amor', 'quiero', 'adoro', 'enamorado', 'enamorada',
                'te amo', 'te quiero', 'me gustas', 'increíble', 'hermoso',
                'hermosa', 'perfecto', 'perfecta', 'feliz contigo', 'juntos'
            ],
            'ruptura': [
                'terminamos', 'acabó', 'no te quiero', 'ya no', 'adiós',
                'dejamos', 'rompimos', 'separamos', 'fin', 'terminar',
                'olvidarte', 'no funciona', 'no puedo más'
            ],
            'confusión': [
                'confundido', 'confundida', 'no sé', 'perdido', 'perdida',
                'dudas', 'inseguro', 'insegura', 'qué hacer', 'no entiendo',
                'complicado', 'difícil', 'no estoy seguro', 'no estoy segura'
            ]
        }
    
    def predict(self, text):
        """Predice la emoción basándose en palabras clave"""
        text_lower = text.lower()
        
        # Contar coincidencias para cada emoción
        scores = {}
        for emotion, keywords in self.keywords.items():
            score = 0
            matched_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
            scores[emotion] = {
                'score': score,
                'keywords': matched_keywords
            }
        
        # Encontrar la emoción con mayor puntuación
        max_score = max(s['score'] for s in scores.values())
        
        if max_score == 0:
            # Si no hay coincidencias, devolver confusión por defecto
            predicted_emotion = 'confusión'
            confidence = 30.0
        else:
            predicted_emotion = max(scores.items(), key=lambda x: x[1]['score'])[0]
            # Calcular confianza basada en el score
            total_score = sum(s['score'] for s in scores.values())
            confidence = (scores[predicted_emotion]['score'] / total_score) * 100
        
        # Crear distribución de probabilidades
        total_score = sum(s['score'] for s in scores.values()) or 1
        probabilities = {
            emotion: round((data['score'] / total_score) * 100, 2) if total_score > 0 else 33.33
            for emotion, data in scores.items()
        }
        
        return {
            'emotion': predicted_emotion,
            'confidence': round(confidence, 2),
            'probabilities': probabilities,
            'matched_keywords': scores[predicted_emotion]['keywords']
        }
