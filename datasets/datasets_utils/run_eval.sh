GROUND_TRUTH="../AIC22_Track1_MTMC_Tracking/algorithm_results/xxx.txt"
PREDICTION="/mnt/LocalDisk1/Projects/AIC21-MTMC/reid/reid-matching/tools/track1.txt"

python ../AIC22_Track1_MTMC_Tracking/eval/eval.py ${GROUND_TRUTH} ${PREDICTION} --dstype test