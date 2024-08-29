#!/bin/bash

# Set base directory
BASE_DIR="code_review_web_app"

# Create directories
mkdir -p $BASE_DIR/templates
mkdir -p $BASE_DIR/static/css
mkdir -p $BASE_DIR/static/js
mkdir -p $BASE_DIR/logs

# Create app.py
cat <<EOL > $BASE_DIR/app.py
from flask import Flask, render_template, jsonify, request
import os
import re

app = Flask(__name__)

# Function to parse the log file and create the call stack tree
def parse_log_file(file_path):
    with open(file_path, 'r') as f:
        logs = []
        for line in f:
            match_push = re.match(r'^(--+>|\s*>)\s*call function (.+?) in (.+?):(\d+)', line)
            if match_push:
                function_name = match_push.group(2)
                file_path = match_push.group(3)
                line_number = int(match_push.group(4))
                logs.append((function_name, file_path, line_number, "push"))
                continue

            match_pop = re.match(r'^(<--+|\s*<)\s*exit function (.+?) in (.+?):(\d+)', line)
            if match_pop:
                function_name = match_pop.group(2)
                file_path = match_pop.group(3)
                line_number = int(match_pop.group(4))
                logs.append((function_name, file_path, line_number, "pop"))
                continue

    return logs

class CallStackNode:
    def __init__(self, function_name, file_path, line_number):
        self.function_name = function_name
        self.file_path = file_path
        self.line_number = line_number
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

class CallStackTree:
    def __init__(self):
        self.root = CallStackNode("root", "", 0)
        self.current_node = self.root

    def push(self, function_name, file_path, line_number):
        new_node = CallStackNode(function_name, file_path, line_number)
        self.current_node.add_child(new_node)
        self.current_node = new_node

    def pop(self):
        if self.current_node != self.root:
            self.current_node = self.current_node.parent

# Global call stack tree to be used in the Flask routes
call_stack_tree = CallStackTree()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_call_stack', methods=['GET'])
def get_call_stack():
    global call_stack_tree
    def build_tree(node):
        children = [build_tree(child) for child in node.children]
        return {
            'function_name': node.function_name,
            'file_path': node.file_path,
            'line_number': node.line_number,
            'children': children
        }
    return jsonify(build_tree(call_stack_tree.root))

@app.route('/get_code_snippet', methods=['GET'])
def get_code_snippet():
    file_path = request.args.get('file_path')
    line_number = int(request.args.get('line_number'))
    context = int(request.args.get('context', 10))

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        start_line = max(0, line_number - context - 1)
        end_line = min(len(lines), line_number + context)
        displayed_lines = lines[start_line:end_line]

        return jsonify({
            'code': ''.join(displayed_lines),
            'highlight_line': line_number - start_line
        })
    except FileNotFoundError:
        return jsonify({
            'error': f"File not found: {file_path}"
        })

def load_log_file():
    log_file_path = os.path.join('logs', 'tracing-test_schedule-20240829_071457.log')
    logs = parse_log_file(log_file_path)

    global call_stack_tree
    for log in logs:
        function_name, file_path, line_number, operation = log
        if operation == "push":
            call_stack_tree.push(function_name, file_path, line_number)
        elif operation == "pop":
            call_stack_tree.pop()

if __name__ == '__main__':
    load_log_file()
    app.run(debug=True)
EOL

# Create index.html
cat <<EOL > $BASE_DIR/templates/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Review App</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
</head>
<body>
    <div class="container">
        <div class="treeview" id="call-stack">
            <h3>Call Stack</h3>
            <ul id="call-stack-root"></ul>
        </div>
        <div class="codeview">
            <h3>Code Snippet</h3>
            <pre id="code-display"></pre>
        </div>
    </div>

    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
EOL

# Create styles.css
cat <<EOL > $BASE_DIR/static/css/styles.css
.container {
    display: flex;
    height: 100vh;
}

.treeview {
    width: 30%;
    border-right: 1px solid #ccc;
    overflow-y: auto;
    padding: 10px;
}

.codeview {
    width: 70%;
    padding: 10px;
    overflow-y: auto;
}

ul {
    list-style-type: none;
    padding-left: 20px;
}

li {
    cursor: pointer;
    margin-bottom: 5px;
}

li:hover {
    background-color: #f0f0f0;
}

pre {
    background-color: #f7f7f7;
    border: 1px solid #ddd;
    padding: 10px;
    font-family: monospace;
}
EOL

# Create main.js
cat <<EOL > $BASE_DIR/static/js/main.js
document.addEventListener('DOMContentLoaded', function () {
    fetch('/get_call_stack')
        .then(response => response.json())
        .then(data => {
            const rootElement = document.getElementById('call-stack-root');
            buildTree(rootElement, data);
        });

    function buildTree(parentElement, nodeData) {
        const li = document.createElement('li');
        li.textContent = nodeData.function_name;
        li.dataset.filePath = nodeData.file_path;
        li.dataset.lineNumber = nodeData.line_number;
        li.addEventListener('click', onNodeClick);

        if (nodeData.children.length > 0) {
            const ul = document.createElement('ul');
            nodeData.children.forEach(child => buildTree(ul, child));
            li.appendChild(ul);
        }

        parentElement.appendChild(li);
    }

    function onNodeClick(event) {
        const filePath = event.target.dataset.filePath;
        const lineNumber = event.target.dataset.lineNumber;

        fetch(`/get_code_snippet?file_path=${encodeURIComponent(filePath)}&line_number=${lineNumber}`)
            .then(response => response.json())
            .then(data => {
                const codeDisplay = document.getElementById('code-display');
                if (data.error) {
                    codeDisplay.textContent = data.error;
                } else {
                    codeDisplay.innerHTML = '';
                    const codeLines = data.code.split('\n');
                    codeLines.forEach((line, index) => {
                        const lineElement = document.createElement('div');
                        if (index + 1 === data.highlight_line) {
                            lineElement.style.backgroundColor = 'yellow';
                        }
                        lineElement.textContent = line;
                        codeDisplay.appendChild(lineElement);
                    });
                }
            });
    }
});
EOL

echo "Project setup complete. Please add your log file to the logs directory."