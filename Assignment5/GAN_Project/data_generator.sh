#!/bin/bash
time=20
for i in  `seq 0 $time`;
do 
	manta manta_genSimSimple.py
done

# cd ./data
# mv "simSimple_10$time" testing
# cd ..

python DataOrganiser.py
