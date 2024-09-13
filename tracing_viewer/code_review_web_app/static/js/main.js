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
        displayCodeButton.addEventListener('click', function (event) {
            event.stopPropagation();

            // Replace the emoji with the counter and increment it on each click
            displayCodeButton.textContent = clickCount;
            clickCount += 1;

            // Fetch the code snippets for the current function and its parent
            const codeDisplayUpper = document.getElementById('code-display-upper');
            const codeDisplayLower = document.getElementById('code-display-lower');

            // Clear previous code but retain the structure
            codeDisplayUpper.innerHTML = '<p>Loading parent function...</p>';
            codeDisplayLower.innerHTML = '<p>Loading current function...</p>';

            // Fetch lower part (current function)
            fetch(`/get_function_code?file_path=${encodeURIComponent(li.dataset.filePath)}&start_line_number=${li.dataset.startLineNumber}&end_line_number=${li.dataset.endLineNumber}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        codeDisplayLower.textContent = data.error;
                    } else {
                        const preElementLower = document.createElement('pre');
                        preElementLower.classList.add('line-numbers');
                        const codeElementLower = document.createElement('code');
                        codeElementLower.classList.add('language-python');
                        codeElementLower.textContent = data.code;
                        preElementLower.appendChild(codeElementLower);
                        codeDisplayLower.innerHTML = ''; // Clear loading message
                        codeDisplayLower.appendChild(preElementLower);
                        Prism.highlightElement(codeElementLower);
                    }
                });

            // Fetch upper part (parent function)
            if (nodeData.parent) {
                fetch(`/get_function_code?file_path=${encodeURIComponent(nodeData.parent.file_path)}&start_line_number=${nodeData.parent.start_line_number}&end_line_number=${nodeData.parent.end_line_number}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            codeDisplayUpper.textContent = data.error;
                        } else {
                            const preElementUpper = document.createElement('pre');
                            preElementUpper.classList.add('line-numbers');
                            const codeElementUpper = document.createElement('code');
                            codeElementUpper.classList.add('language-python');
                            codeElementUpper.textContent = data.code;
                            preElementUpper.appendChild(codeElementUpper);
                            codeDisplayUpper.innerHTML = ''; // Clear loading message
                            codeDisplayUpper.appendChild(preElementUpper);
                            Prism.highlightElement(codeElementUpper);
                        }
                    });
            } else {
                codeDisplayUpper.innerHTML = '<p>No parent function available.</p>';
            }
        });

        li.appendChild(displayCodeButton);

        // Add a border or indentation for clarity
        li.style.marginLeft = '20px';

        if (nodeData.children.length > 0) {
            const ul = document.createElement('ul');
            nodeData.children.forEach(child => {
                child.parent = nodeData; // Set the parent reference
                buildTree(ul, child);
            });
            li.appendChild(ul);
        }

        parentElement.appendChild(li);
    }
});