"""
Below is the code for code_snippet_extractor.py, which extracts a snippet of code from a given file based on the line number provided.

Explanation of the Code

	•	Parameters:
	•	file_path: Path to the source code file from which you want to extract the snippet.
	•	line_number: The specific line number around which the snippet should be extracted (1-based index).
	•	context: The number of lines to include before and after the specified line. This provides context for the snippet.
	•	Error Handling:
	•	The script handles cases where the file is not found (FileNotFoundError) or other general exceptions that might occur during file reading.
	•	Snippet Extraction:
	•	The code calculates the start_line and end_line to ensure they are within the bounds of the file.
	•	It reads the lines from the file and returns the snippet as a single string, combining the lines.
	•	Usage:
	•	You can run the script directly to test the extraction of a code snippet, or you can import the get_code_snippet function into other parts of your application, such as the Flask API, to dynamically retrieve code snippets.

This module is designed to be simple and effective for extracting relevant parts of a source code file, which can then be displayed on a web page when a user clicks on a log entry.
"""

def get_code_snippet(file_path, line_number, context=5):
    """
    Extracts a code snippet from the given file around the specified line number.

    Parameters:
    file_path (str): The path to the source code file.
    line_number (int): The line number around which to extract the code snippet.
    context (int): The number of lines before and after the specified line to include in the snippet.

    Returns:
    str: A string containing the code snippet.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Calculate the start and end lines for the snippet
        start_line = max(0, line_number - context - 1)  # line_number is 1-based index
        end_line = min(len(lines), line_number + context)

        # Extract the snippet
        snippet = lines[start_line:end_line]

        return ''.join(snippet)

    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage:
if __name__ == "__main__":
    file_path = "/root/vescale_prj/veScale/vescale/dtensor/device_mesh.py"
    line_number = 153
    snippet = get_code_snippet(file_path, line_number)
    print(snippet)