#!/bin/bash

# Loop from 2 to 3 with step 0.1
for i in $(LC_ALL=C seq 2.0 0.1 2.5)
do
   # Call your script with i as an argument
   ./get_data.py 500 5000 Data/TrainDataset$i $i 0 32 10 -w 1
done

for i in $(LC_ALL=C seq 2.6 0.1 3.0)
do
   # Call your script with i as an argument
   ./get_data.py 1500 5000 Data/TrainDataset$i $i 0 32 10 -w 1
done