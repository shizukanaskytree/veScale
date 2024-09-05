from flask import Flask, jsonify, render_template, send_from_directory
import os

app = Flask(__name__)

LOGS_DIR = './logs/steps/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/steps/<int:step_number>', methods=['GET'])
def get_step(step_number):
    step_file = f'step_{step_number:03d}.py'
    step_path = os.path.join(LOGS_DIR, step_file)

    if os.path.exists(step_path):
        with open(step_path, 'r') as file:
            step_content = file.read()
        return jsonify({'step_number': step_number, 'content': step_content})
    else:
        return jsonify({'error': 'Step not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5001)