export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
export OUTPUT_DIR='/mnt/petrelfs/yanziang/Ask-Anything/video_chat2/ckpts'
echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=2
# MASTER_NODE='SH-IDCA1404-10-140-54-7'




torchrun  --nnodes=${NNODE} --nproc_per_node=${NUM_GPUS} \
    tasks/train_qformer.py \
    /mnt/petrelfs/yanziang/Ask-Anything/video_chat2/scripts/config_7b_stage1.py \
    output_dir ${OUTPUT_DIR}
# torchrun  --nnodes=${NNODE} --nproc_per_node=${NUM_GPUS} \
#     --rdzv_endpoint=${MASTER_NODE}:10068 \
#     --rdzv_backend=c10d \
#     tasks/train_qformer.py \
#     $(dirname $0)/config_7b_stage1.py \
#     output_dir ${OUTPUT_DIR}
