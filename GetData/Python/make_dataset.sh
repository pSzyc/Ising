#!/bin/bash

# Loop from 2 to 3 with step 0.1
for i in $(LC_NUMERIC=C seq 2 0.1 3)
do
   # Call your script with i as an argument
   ./get_data.py 1500 50000 Data/Data$i $i 0 32 10 -w 1 -s 1
done