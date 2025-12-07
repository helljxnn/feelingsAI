from flask import Flask
from controllers import main_controller

app = Flask(__name__)

# Rutas
app.route('/')(main_controller.index)
app.route('/analyze', methods=['POST'])(main_controller.analyze)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
