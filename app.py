from flask import Flask
from controllers import main_controller

app = Flask(__name__)

# Rutas
app.add_url_rule('/', 'index', main_controller.index)
app.add_url_rule('/analyze', 'analyze', main_controller.analyze, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
