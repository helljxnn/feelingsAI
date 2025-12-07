from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class EmotionClassifier:
    def __init__(self, model_path='model_final'):
        self.model_path = model_path
        self.hf_model = 'helljxnn/feelingsai-emotion-classifier'
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = {0: 'Enamoramiento', 1: 'Ruptura', 2: 'Confusión'}
        
    def load_model(self):
        """Carga el modelo y tokenizer"""
        try:
            # Intenta cargar el modelo local primero
            if os.path.exists(self.model_path):
                print(f"Cargando modelo local desde {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            else:
                # Si no existe localmente, descarga desde Hugging Face
                print(f"Descargando modelo desde Hugging Face: {self.hf_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.hf_model)
                print("Modelo descargado exitosamente")
            
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def predict(self, text):
        """Predice la emoción del texto"""
        if not self.model or not self.tokenizer:
            raise Exception("Modelo no cargado")
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                               max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'emotion': self.labels[predicted_class],
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                self.labels[i]: round(probabilities[0][i].item() * 100, 2) 
                for i in range(len(self.labels))
            }
        }
