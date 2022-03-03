#!/bin/bash
# THUCNews is a Chinese news corpus launched by the Natural Language Processing Laboratory of Tsinghua University.
# THUCNews is publicly available in http://thuctc.thunlp.org/

path=$1/THUCNews
txts="train.txt dev.txt test.txt class.txt"
mkdir -p $path
for i in $txts;
do 
    wget "https://gcrblob.blob.core.windows.net/downloads/THUCNews/data/$i" -O "$path/$i"; 
done
