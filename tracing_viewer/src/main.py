"""
main.py 是你的应用程序的入口点，负责启动整个应用程序。以下是一个简单的 main.py 示例代码，它使用 Flask 框架来启动一个 Web 服务器，并为日志文件的解析和代码片段的展示提供后端服务。

Flask 应用:
	•	app = Flask(__name__) 初始化 Flask 应用。
	•	@app.route('/') 定义主页路由，渲染 index.html 模板，并传递解析后的日志条目。
	•	@app.route('/get_code_snippet', methods=['GET']) 定义一个 API 端点，用于根据文件路径和行号返回相应的代码片段。
	•	日志解析:
	•	LOG_FILE_PATH 是你的日志文件路径。
	•	log_entries = parse_log_file(LOG_FILE_PATH) 解析日志文件，并将结果缓存，以便在渲染页面时使用。
	•	代码片段提取:
	•	get_code_snippet(file, line) 使用 code_snippet_extractor.py 中定义的函数，根据请求参数提取代码片段并返回 JSON 响应。
	•	运行应用:
	•	app.run(debug=True, host='0.0.0.0', port=5000) 启动 Flask 开发服务器，监听所有网络接口上的 5000 端口。

依赖文件

	•	log_parser.py 和 code_snippet_extractor.py 是你在 src/ 目录下定义的模块，用于解析日志文件和提取代码片段。你需要确保这些模块中的功能能够正确读取和处理文件，并返回期望的结果。

这样，运行 main.py 后，你可以通过访问 http://localhost:5000/ 来查看解析后的日志内容，并在点击相应条目时展示对应的代码片段。


The application you've built can dynamically load code snippets and log entries due to the way it interacts with the client (browser) and server (Flask application) using AJAX (Asynchronous JavaScript and XML). Here’s a breakdown of how this dynamic loading works:

### Dynamic Loading of Code Snippets

1. **AJAX Request to the Server**:
   - When a user clicks on the "Show Code" button in the web interface, the JavaScript in `main.js` captures this event and sends an asynchronous request (AJAX) to the server using the Fetch API.
   - The request is sent to the `/get_code_snippet` endpoint, including the file path and line number as query parameters.

2. **Server Processing**:
   - The Flask application receives the request at the `/get_code_snippet` route.
   - The server extracts the `file` and `line` parameters from the request, validates them, and uses the `get_code_snippet` function to retrieve the relevant code snippet from the specified file.

3. **Returning the Snippet**:
   - Once the server has retrieved the code snippet, it returns this data as a JSON response to the client.

4. **Updating the Web Page**:
   - The JavaScript on the client side processes the JSON response and dynamically inserts the retrieved code snippet into the HTML, displaying it to the user without requiring a page refresh.

### Dynamic Loading of Log Entries

1. **Initial Load**:
   - When the user first loads the webpage, only the first portion of the log file (e.g., the first 100 lines) is loaded and rendered. This keeps the initial load time short and the page responsive.

2. **AJAX Request for More Entries**:
   - As the user scrolls down or clicks the "Load More" button, another AJAX request is made to the server, specifically to the `/load-log` endpoint.
   - This request includes parameters for `start_line` and `end_line` to specify the next chunk of the log file to load.

3. **Server Processing**:
   - The Flask application processes this request in the `/load-log` route. It uses the `parse_log_file` function to extract the specified portion of the log file and converts it into a list of entries.
   - These entries are then returned as a JSON response.

4. **Appending New Entries**:
   - The client-side JavaScript receives this JSON response and dynamically adds the new log entries to the existing list on the page. This operation is smooth and doesn't require reloading the entire page.

### Why Dynamic Loading is Effective

- **Performance**: By only loading the data the user is currently viewing (or requesting), you avoid the performance bottlenecks associated with loading large log files or large amounts of data all at once. This reduces memory usage and initial load times.

- **User Experience**: Users experience a responsive interface, where they can interact with the application and request more data on-demand without waiting for unnecessary data to load.

- **Scalability**: The approach scales well with large log files because it processes and delivers data in manageable chunks rather than overwhelming the server or the client with large amounts of data.

In summary, the dynamic nature of this application comes from its ability to fetch and display only the relevant pieces of information on-demand. This is achieved through the combination of client-side JavaScript to handle user interactions and server-side Flask routes to process and return the requested data.
"""


from flask import Flask, render_template, request, jsonify
from log_parser import parse_log_file
from code_snippet_extractor import get_code_snippet
import os

app = Flask(__name__)

# 设置日志文件的路径
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'logs', 'tracing-finetune_4D-20240821_144339.log')

@app.route('/')
def index():
    # Initial load of log entries (e.g., the first 100 lines)
    initial_entries = parse_log_file(LOG_FILE_PATH, start_line=0, end_line=100)
    return render_template('index.html', log_entries=initial_entries)

@app.route('/load-log', methods=['GET'])
def load_log():
    """
    API endpoint to dynamically load a portion of the log file based on user request.
    """
    start_line = int(request.args.get('start', 0))
    end_line = int(request.args.get('end', 100))  # Default to loading 100 lines at a time

    log_entries = parse_log_file(LOG_FILE_PATH, start_line=start_line, end_line=end_line)
    return jsonify(log_entries)

@app.route('/get_code_snippet', methods=['GET'])
def serve_code_snippet():
    """
    API endpoint to serve a code snippet based on file path and line number.
    """
    file = request.args.get('file')
    line = request.args.get('line')

    if not file or not line:
        return jsonify({'snippet': 'Error: Missing file or line number'}), 400

    try:
        line = int(line)
    except ValueError:
        return jsonify({'snippet': 'Error: Invalid line number'}), 400

    snippet = get_code_snippet(file, line)
    if snippet:
        return jsonify({'snippet': snippet})
    else:
        return jsonify({'snippet': 'Error: Could not retrieve snippet'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

