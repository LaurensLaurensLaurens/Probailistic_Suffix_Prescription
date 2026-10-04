#!/bin/bash
cd data

bpic_2012="https://data.4tu.nl/ndownloader/items/533f66a4-8911-4ac7-8612-1235d65d1f37/versions/1"
wget -O bpic_2012.zip "$bpic_2012"
unzip -o bpic_2012.zip
rm DATA.xml
gunzip -f BPI_Challenge_2012.xes.gz
rm bpic_2012.zip

bpic_2013="https://data.4tu.nl/ndownloader/items/0fc5c579-e544-4fab-9143-fab1f5192432/versions/1"
wget -O bpic_2013.zip "$bpic_2013"
unzip -o bpic_2013.zip
rm DATA.xml
gunzip -f BPI_Challenge_2013_incidents.xes.gz
rm bpic_2013.zip

bpic_2014="https://data.4tu.nl/ndownloader/items/657fb1d6-b4c2-4adc-ba48-ed25bf313025/versions/1"
wget -O bpic_2014.zip "$bpic_2014"
unzip -o bpic_2014.zip
rm bpic_2014.zip

bpic_2017="https://data.4tu.nl/ndownloader/items/34c3f44b-3101-4ea9-8281-e38905c68b8d/versions/1"
wget -O bpic_2017.zip "$bpic_2017"
unzip -o bpic_2017.zip
rm DATA.xml
gunzip -f 'BPI Challenge 2017.xes.gz'
rm bpic_2017.zip

bpic_2018="https://data.4tu.nl/ndownloader/items/443451fd-d38a-4464-88b4-0fc641552632/versions/1"
wget -O bpic_2018.zip "$bpic_2018"
unzip -o bpic_2018.zip
rm DATA.xml
gunzip -f 'BPI Challenge 2018.xes.gz'
rm bpic_2018.zip

bpic_2019="https://data.4tu.nl/ndownloader/items/35ed7122-966a-484e-a0e1-749b64e3366d/versions/1"
wget -O bpic_2019.zip "$bpic_2019"
unzip -o bpic_2019.zip
rm bpic_2019.zip

bpic_2020_dd="https://data.4tu.nl/ndownloader/items/6a0a26d2-82d0-4018-b1cd-89afb0e8627f/versions/1"
wget -O bpic_2020_dd.zip "$bpic_2020_dd"
unzip -o bpic_2020_dd.zip
rm README.txt
gunzip -f 'DomesticDeclarations.xes.gz'
rm bpic_2020_dd.zip

PCR_event_log="https://lehre.bpm.in.tum.de/~kunkler/pcr_event_log.csv"
wget -O pcr_event_log.csv "$PCR_event_log"
