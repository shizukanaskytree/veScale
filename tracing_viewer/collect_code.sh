#!/bin/bash

# bash collect_code.sh /path/to/target_directory

# Directory to search
TARGET_DIR=${1:-"."}

# Output file
OUTPUT_FILE="collected_code.txt"

# Ensure the output file is empty
> $OUTPUT_FILE

# Function to collect text from all files except .log files
collect_code() {
    local dir="$1"
    for file in "$dir"/*; do
        if [ -d "$file" ]; then
            collect_code "$file" # Recursive call if it's a directory
        elif [ -f "$file" ] && [[ "$file" != *.log ]]; then
            echo "- $(basename "$file")" >> $OUTPUT_FILE
            echo "" >> $OUTPUT_FILE
            cat "$file" >> $OUTPUT_FILE
            echo "" >> $OUTPUT_FILE
        fi
    done
}

# Start collecting code
collect_code "$TARGET_DIR"

echo "Code collection completed. Output saved to $OUTPUT_FILE"