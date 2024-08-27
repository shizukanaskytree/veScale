"""
Below is an example of log_parser.py, which will parse the tracing-finetune_4D-20240821_144339.log file to extract the call and exit function entries.

Explanation of the Code

	•	Regular Expression Pattern: The pattern is designed to match both call and exit lines in the log. It identifies the function name, file path, and line number, along with the indentation level.
	•	Indentation Level: The indentation level is calculated based on the number of - characters before the > symbol. This helps in reconstructing the call stack visually.
	•	Output: Each entry in the parsed log is stored as a dictionary containing the type (call or exit), function name, file path, line number, and indentation level.
	•	Usage: You can run the script directly to parse the log file and print the parsed entries, or you can import the parse_log_file function into other parts of your application.

This script will give you a structured way to handle your log data, which can then be used in your Flask application to display information on the web page.
"""

import re

def parse_log_file(log_file_path, start_line=0, end_line=None):
    """
    Parses a specific range of lines in a log file to extract function call and exit events with the code between them.

    Parameters:
    log_file_path (str): Path to the log file.
    start_line (int): The line number to start reading from.
    end_line (int): The line number to stop reading at.

    Returns:
    list: A list of dictionaries, each representing a function call with its corresponding exit and the code in between.
    """
    log_entries = []

    call_pattern = r'(--+>)\s*call function (\S+) in (\S+):(\d+)'
    exit_pattern = r'(<--+)\s*exit function (\S+) in (\S+):(\d+)'

    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()[start_line:end_line]

    current_call = None
    for i, line in enumerate(lines, start=start_line):
        call_match = re.search(call_pattern, line)
        exit_match = re.search(exit_pattern, line)

        if call_match:
            current_call = {
                'type': 'call',
                'function': call_match.group(2),
                'file': call_match.group(3),
                'line': int(call_match.group(4)),
                'indentation': len(call_match.group(1)) // 2,
                'code': []
            }
        elif exit_match and current_call:
            if current_call['function'] == exit_match.group(2):
                current_call['code'].extend(lines[current_call['line']-start_line:i-start_line])
                current_call['code'] = ''.join(current_call['code'])
                log_entries.append(current_call)
                log_entries.append({
                    'type': 'exit',
                    'function': exit_match.group(2),
                    'file': exit_match.group(3),
                    'line': int(exit_match.group(4)),
                    'indentation': len(exit_match.group(1)) // 2
                })
                current_call = None

    return log_entries


# Example usage:
if __name__ == "__main__":
    log_file_path = "/root/vescale_prj/veScale/tracing_viewer/logs/tracing-finetune_4D-20240821_144339.log"
    entries = parse_log_file(log_file_path)
    for entry in entries:
        print(entry)
