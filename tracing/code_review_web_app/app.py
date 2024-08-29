from flask import Flask, render_template, jsonify, request
import os
import re

app = Flask(__name__)

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
    def __init__(self, function_name, file_path, start_line_number):
        self.function_name = function_name
        self.file_path = file_path
        self.start_line_number = start_line_number
        self.end_line_number = None
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def set_end_line_number(self, end_line_number):
        self.end_line_number = end_line_number

class CallStackTree:
    def __init__(self):
        self.root = CallStackNode("root", "", 0)
        self.current_node = self.root

    def push(self, function_name, file_path, start_line_number):
        new_node = CallStackNode(function_name, file_path, start_line_number)
        self.current_node.add_child(new_node)
        self.current_node = new_node

    def pop(self, end_line_number):
        if self.current_node != self.root:
            self.current_node.set_end_line_number(end_line_number)
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
            'start_line_number': node.start_line_number,
            'end_line_number': node.end_line_number,
            'children': children
        }
    return jsonify(build_tree(call_stack_tree.root))

@app.route('/get_function_code', methods=['GET'])
def get_function_code():
    file_path = request.args.get('file_path')
    start_line_number = int(request.args.get('start_line_number'))
    end_line_number = request.args.get('end_line_number')

    # If the end_line_number is missing or invalid, set a default value
    if not end_line_number or end_line_number == 'null':
        end_line_number = start_line_number + 20  # Arbitrary number of lines to display if end_line_number is missing
    else:
        end_line_number = int(end_line_number)

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        displayed_lines = lines[start_line_number - 1:end_line_number]
        numbered_lines = [f"{i + start_line_number:4d}: {line}" for i, line in enumerate(displayed_lines)]

        return jsonify({
            'code': ''.join(numbered_lines)
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
            call_stack_tree.pop(line_number)

if __name__ == '__main__':
    load_log_file()
    app.run(debug=True)