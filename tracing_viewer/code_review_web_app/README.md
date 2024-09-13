# How to start?

```bash
root@f78e8e970a17:~/vescale_prj/veScale/tracing/code_review_web_app# python app.py

 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 104-502-987
127.0.0.1 - - [29/Aug/2024 10:03:07] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [29/Aug/2024 10:03:11] "GET /static/js/main.js HTTP/1.1" 304 -
127.0.0.1 - - [29/Aug/2024 10:03:11] "GET /static/js/resize.js HTTP/1.1" 304 -
127.0.0.1 - - [29/Aug/2024 10:03:11] "GET /static/css/styles.css HTTP/1.1" 200 -
127.0.0.1 - - [29/Aug/2024 10:03:12] "GET /get_call_stack HTTP/1.1" 200 -
```


# Font size of the tree view content

The font size of the tree view content is primarily controlled by the CSS `font-size` property applied to the elements within the tree view. Specifically, the following CSS rules control the font size:

1. **Treeview Container**:
   - The `.treeview` class defines the overall font size for the entire tree view. Setting the `font-size` here will apply to all text within the tree view unless overridden by more specific rules.

   ```css
   .treeview {
       font-size: 10px; /* Controls the general font size of the tree view content */
   }
   ```

2. **List Items (`li`)**:
   - The `li` elements represent individual nodes in the tree. If you set a `font-size` here, it will directly control the font size of each item in the tree.

   ```css
   li {
       font-size: 10px; /* Controls the font size of each list item in the tree */
   }
   ```

3. **Function Name Span**:
   - The `.function-name` class is used for the span that displays the function name. If this is set, it will specifically control the font size of the function names within each list item.

   ```css
   li .function-name {
       font-size: 10px; /* Controls the font size of function names */
   }
   ```

4. **Buttons**:
   - The `button.display-code` class controls the font size of the buttons within the tree view. Adjusting this will change the size of the text inside the buttons.

   ```css
   button.display-code {
       font-size: 10px; /* Controls the font size of the buttons */
   }
   ```

### Summary of Control Points:
- **Overall Tree View**: `.treeview { font-size: ...; }`
- **Individual Tree Items**: `li { font-size: ...; }`
- **Function Names**: `li .function-name { font-size: ...; }`
- **Buttons**: `button.display-code { font-size: ...; }`

By adjusting these CSS properties, you can precisely control the font size of different parts of the tree view content. If you want to reduce the overall font size, start with the `.treeview` class. If you need finer control, adjust the `li`, `.function-name`, and `button.display-code` classes as needed.