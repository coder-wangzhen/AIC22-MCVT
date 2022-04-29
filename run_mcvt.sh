MCMT_CONFIG_FILE="aic_all.yml"

#### MOT. ####
cd mot
bash run_mot.sh ${MCMT_CONFIG_FILE}
wait

#### Get results. ####
cd ../matching
python trajectory_fusion.py ${MCMT_CONFIG_FILE} && python sub_cluster.py ${MCMT_CONFIG_FILE} && python gen_res.py ${MCMT_CONFIG_FILE}
