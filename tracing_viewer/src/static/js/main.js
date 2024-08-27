document.addEventListener("DOMContentLoaded", function() {
    let currentStartLine = 100;  // Assuming the initial load is 0-100 lines
    const loadSize = 100;  // Number of lines to load at a time

    // Function to fetch and display code snippets
    function fetchCodeSnippet(button) {
        const file = button.getAttribute("data-file");
        const line = button.getAttribute("data-line");
        const codeSnippetContainer = button.closest(".log-entry").querySelector(".code-snippet");

        // Toggle the display of the code snippet area
        if (codeSnippetContainer.style.display === "none" || !codeSnippetContainer.style.display) {
            codeSnippetContainer.style.display = "block"; // Expand to show the code
            fetch(`/get_code_snippet?file=${encodeURIComponent(file)}&line=${line}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    codeSnippetContainer.textContent = data.snippet || "No snippet found";
                })
                .catch(error => {
                    console.error("Error fetching code snippet:", error);
                    codeSnippetContainer.textContent = "Error fetching code snippet";
                });
        } else {
            codeSnippetContainer.style.display = "none"; // Collapse if already shown
        }
    }

    // Add event listeners to existing "Show Code" buttons
    document.querySelectorAll(".show-code-btn").forEach(button => {
        button.addEventListener("click", function() {
            fetchCodeSnippet(button);
        });
    });

    // Function to load more log entries
    function loadMoreLogEntries() {
        fetch(`/load-log?start=${currentStartLine}&end=${currentStartLine + loadSize}`)
            .then(response => response.json())
            .then(data => {
                const logContainer = document.getElementById("log-container");
                data.forEach(entry => {
                    const entryDiv = document.createElement("div");
                    entryDiv.className = "log-entry";
                    entryDiv.setAttribute("data-file", entry.file);
                    entryDiv.setAttribute("data-line", entry.line);

                    const entryContent = `
                        <span style="margin-left: ${entry.indentation}em;">
                            ${entry.type === 'call' ? `➔ call function <strong>${entry.function}</strong> in ${entry.file}:${entry.line}` : `⇐ exit function <strong>${entry.function}</strong> in ${entry.file}</strong>:${entry.line}`}
                            ${entry.type === 'call' ? `<button class="show-code-btn" data-file="${entry.file}" data-line="${entry.line}">Show Code</button>` : ''}
                        </span>
                        <div class="code-snippet" style="display: none; margin-left: ${entry.indentation + 2}em;"></div>
                    `;
                    entryDiv.innerHTML = entryContent;
                    logContainer.appendChild(entryDiv);

                    // Add event listener to the newly created "Show Code" button
                    if (entry.type === 'call') {
                        entryDiv.querySelector(".show-code-btn").addEventListener("click", function() {
                            fetchCodeSnippet(this);
                        });
                    }
                });
                currentStartLine += loadSize; // Update the start line for the next load
            })
            .catch(error => {
                console.error("Error loading more log entries:", error);
            });
    }

    // Add event listener to the "Load More" button
    document.getElementById("load-more-btn").addEventListener("click", loadMoreLogEntries);
});