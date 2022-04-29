#!/bin/bash
cd tool
python pre_process.py

cd ..
if [ ! -d "./build" ];then
    mkdir build
else
    rm -rf ./build/*
fi
cd build && cmake .. && make -j4
./city_tracker

cd ../tool
python post_precess.py

seqs=(c041 c042 c043 c044 c045 c046)
# seqs=(c042)

TrackOneSeq(){
    seq=$1
    config=$2
    echo save_sot $seq with ${config}
    python save_mot.py ${seq} pp ${config}
}

for seq in ${seqs[@]}
do 
    TrackOneSeq ${seq} $1 &
done
wait
