seqs=(c041 c042 c043 c044 c045 c046)
for seq in ${seqs[@]}
do
    python detect2img.py --name ${seq} --weights ./weights/yolov5x6.pt --conf-thres 0.1 --agnostic --save-txt --save-conf --img-size 1280 --classes 2 5 7 --cfg_file $1
done
wait
