document.addEventListener('DOMContentLoaded', function () {
    fetch('/get_call_stack')
        .then(response => response.json())
        .then(data => {
            const rootElement = document.getElementById('call-stack-root');
            buildTree(rootElement, data);
        });

    function buildTree(parentElement, nodeData) {
        const li = document.createElement('li');

        const functionNameSpan = document.createElement('span');
        functionNameSpan.textContent = `${nodeData.function_name} (${nodeData.file_path}:${nodeData.start_line_number}-${nodeData.end_line_number || '...'})`;
        functionNameSpan.classList.add('function-name');
        li.appendChild(functionNameSpan);

        li.dataset.filePath = nodeData.file_path;
        li.dataset.startLineNumber = nodeData.start_line_number;
        li.dataset.endLineNumber = nodeData.end_line_number;
        li.title = `${nodeData.function_name} in ${nodeData.file_path}:${nodeData.start_line_number}-${nodeData.end_line_number || '...'}`;

        // Initialize counter
        let clickCount = 1;

        const displayCodeButton = document.createElement('button');
        displayCodeButton.textContent = '🐣';
        displayCodeButton.classList.add('display-code');
        displayCodeButton.addEventListener('click', function(event) {
            event.stopPropagation();

            // Replace the emoji with the counter and increment it on each click
            displayCodeButton.textContent = clickCount;
            clickCount += 1;

            // Fetch the code snippet using the API
            fetch(`/get_function_code?file_path=${encodeURIComponent(li.dataset.filePath)}&start_line_number=${li.dataset.startLineNumber}&end_line_number=${li.dataset.endLineNumber}`)
                .then(response => response.json())
                .then(data => {
                    const codeDisplay = document.getElementById('code-display');
                    codeDisplay.innerHTML = '';  // Clear previous code

                    if (data.error) {
                        codeDisplay.textContent = data.error;
                    } else {
                        // Create a pre element with syntax highlighting class and set the code inside it
                        const preElement = document.createElement('pre');
                        preElement.classList.add('line-numbers'); // Optional: Adds line numbers
                        const codeElement = document.createElement('code');
                        codeElement.classList.add('language-python'); // Specify the language for Prism.js
                        codeElement.textContent = data.code;

                        preElement.appendChild(codeElement);
                        codeDisplay.appendChild(preElement);

                        // Re-run Prism to apply syntax highlighting
                        Prism.highlightElement(codeElement);
                    }
                });
        });
        li.appendChild(displayCodeButton);

        // Add a border or indentation for clarity
        li.style.marginLeft = '20px';

        if (nodeData.children.length > 0) {
            const ul = document.createElement('ul');
            nodeData.children.forEach(child => buildTree(ul, child));
            li.appendChild(ul);
        }

        parentElement.appendChild(li);
    }
});