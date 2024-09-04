# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

import re
import ast
import os

class CallContext:
    def __init__(self, file_path, function_name, line_number, variables):
        self.file_path = file_path
        self.function_name = function_name
        self.line_number = line_number
        self.variables = variables  # Keep this as a list of dicts

class ExecutionTracker:
    def __init__(self):
        self.context_stack = []
        self.current_context = None
        self.line_variables = {}  # Stores variables for each line

    def push_context(self, file_path, function_name, line_number, variables):
        if self.current_context:
            self.context_stack.append(self.current_context)
        self.current_context = CallContext(file_path, function_name, line_number, [variables])

    def pop_context(self):
        if self.context_stack:
            self.current_context = self.context_stack.pop()

    def update_context(self, new_file_path=None, new_function_name=None, new_line_number=None, new_variables=None):
        if self.current_context:
            if new_file_path:
                self.current_context.file_path = new_file_path
            if new_function_name:
                self.current_context.function_name = new_function_name
            if new_line_number:
                self.current_context.line_number = new_line_number
            if new_variables:
                self.current_context.variables.append(new_variables)  # Append new variables

    def store_variable(self, line_number, new_variables):
        """Store variables for a specific line number, keeping track of previously added variables."""
        if line_number not in self.line_variables:
            self.line_variables[line_number] = []  # Initialize if not already stored

        # Append new variables while preserving previous ones
        self.line_variables[line_number].append(new_variables)

    def get_variables_for_line(self, line_number):
        """Retrieve all the variables stored for a specific line number."""
        return self.line_variables.get(line_number, [])

    def get_all_variables(self):
        """Retrieve all stored variables for all lines, sorted by line number."""
        all_vars = []
        for line_number in sorted(self.line_variables.keys()):
            vars_for_line = {
                "line_number": line_number,
                "variables": self.line_variables[line_number]
            }
            all_vars.append(vars_for_line)
        return all_vars

    def get_current_context(self):
        return self.current_context

class LogParser:
    def __init__(self, log_file_path, output_dir='./logs/steps'):
        self.log_file_path = log_file_path
        self.output_dir = output_dir
        self.parsed_steps = []
        self.current_step = -1
        self.execution_tracker = ExecutionTracker()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def parse_log(self):
        with open(self.log_file_path, 'r') as file:
            log_lines = file.readlines()

        current_line = None

        for line in log_lines:
            stripped_line = line.strip()

            # Function call identification
            if ">>> Call to" in stripped_line:
                call_match = re.search(r'>>> Call to (.*) in File "(.*)", line (\d+)', stripped_line)
                if call_match:
                    function_name = call_match.group(1)
                    file_path = call_match.group(2)
                    line_number = int(call_match.group(3))
                    self.execution_tracker.push_context(file_path, function_name, line_number, {})
                    self.parsed_steps.append({
                        'step_type': 'function_call',
                        'function_name': function_name,
                        'file_path': file_path,
                        'line_number': line_number,
                        'variables': []
                    })
                continue

            # Code line tracking
            code_line_match = re.match(r"(\d{2}:\d{2}:\d{2}\.\d{2})\s+(\d+)\s+\|\s+(.*)", stripped_line)
            if code_line_match:
                timestamp = code_line_match.group(1)
                line_number = int(code_line_match.group(2))
                code = code_line_match.group(3)
                current_line = {
                    'step_type': 'code_line',
                    'timestamp': timestamp,
                    'line_number': line_number,
                    'code': code,
                    'variables': []
                }
                self.parsed_steps.append(current_line)
                self.write_log_file(current_line)
                continue

            # Variable assignments
            variable_line_match = re.match(r"(\d{2}:\d{2}:\d{2}\.\d{2})\s*\.*\s*(\S+)\s*=\s*(.*)", stripped_line)
            if variable_line_match:
                var_timestamp = variable_line_match.group(1)
                variable_name = variable_line_match.group(2)
                variable_value = variable_line_match.group(3)
                if current_line:
                    # Store the variables incrementally for this line number
                    self.execution_tracker.store_variable(current_line['line_number'], {variable_name: variable_value})

                    # Update parsed_steps with new variables for immediate display
                    current_line['variables'].append({variable_name: variable_value})
                    self.write_log_file(current_line)

            # Return statement handling
            if "<<< Return value from" in stripped_line:
                return_match = re.search(r'<<< Return value from (.*): (.*)', stripped_line)
                if return_match:
                    function_name = return_match.group(1)
                    return_value = return_match.group(2)

                    # Update the current context with the return value
                    self.execution_tracker.update_context(new_variables={"return_value": return_value})

                    # Pop the context stack to revert to the previous function
                    self.execution_tracker.pop_context()

                    # Update parsed steps to include the return statement
                    self.parsed_steps.append({
                        'step_type': 'function_return',
                        'function_name': function_name,
                        'return_value': return_value
                    })
                    self.write_log_file({
                        'step_type': 'function_return',
                        'function_name': function_name,
                        'return_value': return_value
                    })

        # After parsing, reset the file path in parsed_steps to the current context
        current_context = self.execution_tracker.get_current_context()
        if current_context:
            self.parsed_steps[-1]['file_path'] = current_context.file_path

    def get_next_step(self):
        # Advance to the next step in the parsed output
        self.current_step += 1
        if self.current_step < len(self.parsed_steps):
            return self.parsed_steps[self.current_step]
        else:
            return None

    def write_log_file(self, step):
        current_context = self.execution_tracker.get_current_context()
        step_index = len(os.listdir(self.output_dir)) + 1
        file_name = f'step_{step_index:03d}.log'
        file_path = os.path.join(self.output_dir, file_name)

        with open(file_path, 'w') as log_file:
            # Write the full function code
            if current_context:
                function_code, starting_line_number = extract_function_from_file_with_line_numbers(current_context.file_path, current_context.line_number)

                for line_no, line_content in enumerate(function_code.splitlines(), start=starting_line_number):
                    # Mark the line being executed with "=>"
                    if step['step_type'] == 'code_line' and step['line_number'] == line_no:
                        log_file.write(f"=> {line_no:4d}: {line_content}\n")
                    else:
                        log_file.write(f"   {line_no:4d}: {line_content}\n")

                    # Insert preserved and new variables for all executed lines up to now
                    all_vars_for_this_line = self.execution_tracker.get_variables_for_line(line_no)
                    if all_vars_for_this_line:
                        log_file.write("            -----\n")
                        for vars_at_step in all_vars_for_this_line:
                            for name, value in vars_at_step.items():
                                log_file.write(f"            {name} = {value}\n")
                        log_file.write("            -----\n")
                log_file.write("=" * 80 + '\n')

            # Write return statement
            if step['step_type'] == 'function_return':
                log_file.write(f"Return Statement:\n  Function Name: {step['function_name']}\n  Return Value: {step['return_value']}\n")
                log_file.write("=" * 80 + '\n')


"""
To ensure that extract_function_from_file_with_line_numbers preserves all code,
including comments and empty lines, we need to avoid relying on ast.parse() and
ast.unparse(), as they do not retain comments or blank lines. Instead, we should
extract the function’s code directly by reading the source code as plain text and
identifying the function boundaries using line numbers.
"""
# def extract_function_from_file_with_line_numbers(file_path, line_number):
#     with open(file_path, 'r') as file:
#         source = file.read()

#     # Parse the source code into an AST
#     tree = ast.parse(source)

#     # Extract all function definitions
#     functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

#     for func in functions:
#         if func.lineno <= line_number < (func.end_lineno if hasattr(func, 'end_lineno') else float('inf')):
#             function_code = ast.unparse(func)
#             return function_code, func.lineno

#     return None, None


def extract_function_from_file_with_line_numbers(file_path, line_number):
    with open(file_path, 'r') as file:
        source_lines = file.readlines()

    # Start searching for the function starting at the provided line_number
    func_start = None
    func_end = None
    indent_level = None
    for i, line in enumerate(source_lines):
        if i + 1 == line_number and 'def ' in line:  # Start of the function
            func_start = i
            indent_level = len(line) - len(line.lstrip())  # Track the indentation level of the function
            break

    if func_start is None:
        return None, None

    # Find where the function ends based on indentation
    for i, line in enumerate(source_lines[func_start + 1:], start=func_start + 1):
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= indent_level and line.strip() and not line.lstrip().startswith('#'):
            # Function ends when we encounter a line with less or equal indentation
            func_end = i
            break

    # If no function end found, assume it goes till the end of the file
    if func_end is None:
        func_end = len(source_lines)

    # Extract the lines of the function, including comments and empty lines
    function_code = ''.join(source_lines[func_start:func_end])

    return function_code, func_start + 1


if __name__ == "__main__":
    # Test with the sample.log file
    log_file_path = './logs/sample.log'
    output_dir = './logs/steps'

    log_parser = LogParser(log_file_path, output_dir)
    log_parser.parse_log()

    print(f"Log files generated in directory: {output_dir}")