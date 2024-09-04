#!/bin/bash

# bash collect_code.sh /path/to/target_directory

# Directory to search
TARGET_DIR=${1:-"."}

# Output file
OUTPUT_FILE="code.txt"

# Ensure the output file is empty
> $OUTPUT_FILE

# Function to collect text from all files except .log files and specific excluded files
collect_code() {
    local dir="$1"
    for file in "$dir"/*; do
        if [ -d "$file" ]; then
            # Exclude __pycache__ directories and recursive call if it's a directory
            if [[ "$(basename "$file")" != "__pycache__" ]]; then
                collect_code "$file"
            fi
        elif [ -f "$file" ] && [[ "$file" != *.md ]] && [[ "$file" != *.log ]] && [[ "$file" != *.pyc ]] && [[ "$file" != "$OUTPUT_FILE" ]] && [[ "$(basename "$file")" != "collected_code.txt" ]]; then
            # Skip specific binary file, setup_project.sh, and collect_code.sh
            if [[ "$file" != "tracing_tutor/__pycache__/log_parser.cpython-310.pyc" ]] && [[ "$(basename "$file")" != "setup_project.sh" ]] && [[ "$(basename "$file")" != "collect_code.sh" ]]; then
                echo "- $(basename "$file")" >> $OUTPUT_FILE
                echo "" >> $OUTPUT_FILE
                cat "$file" >> $OUTPUT_FILE
                echo "" >> $OUTPUT_FILE
            fi
        fi
    done
}

# Start collecting code
collect_code "$TARGET_DIR"

echo "Code collection completed. Output saved to $OUTPUT_FILE"