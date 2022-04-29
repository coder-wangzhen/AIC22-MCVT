MCMT_CONFIG_FILE="aic_all.yml"
#### Run Detector.####
cd detector/
python gen_images_aic.py ${MCMT_CONFIG_FILE}

cd yolov5_2022/
bash gen_det.sh ${MCMT_CONFIG_FILE}

#### Extract reid feautres.####
cd ../../reid/
python extract_image_feat.py "aic_reid1.yml"
python extract_image_feat.py "aic_reid2.yml"
python extract_image_feat.py "aic_reid3.yml"
python merge_reid_feat.py ${MCMT_CONFIG_FILE}

#### MOT. ####
cd ../mot
bash run_mot.sh ${MCMT_CONFIG_FILE}
wait

#### Get results. ####
cd ../matching
python trajectory_fusion.py ${MCMT_CONFIG_FILE} && python sub_cluster.py ${MCMT_CONFIG_FILE} && python gen_res.py ${MCMT_CONFIG_FILE}

