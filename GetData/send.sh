#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 input_directory output_directory"
  exit 1
fi

# Assign input and output directories to variables
INPUT_DIR=$1
OUTPUT_DIR=$2

# Run the Python script to convert to TFRecord
python3 to_tfrecord.py "$INPUT_DIR" "$OUTPUT_DIR"

# Copy the output directory to the remote server using scp
scp -r "$OUTPUT_DIR" ps440977@tempac.fuw.edu.pl:~/copy/

echo "Conversion and copying completed successfully!"
