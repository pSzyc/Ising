#!/bin/bash

# Loop from 2 to 3 with step 0.1
for i in $(LC_NUMERIC=C seq 2.32 0.02 2.4)
do
   # Call your script with i as an argument
   ./get_data.py 1000 50000 Data/Data$i $i 0 32 10 -w 1
done