export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=4
NUM_GPUS=8
MASTER_NODE='SH-IDC1-10-140-1-1'

torchrun  --nnodes=${NNODE} --nproc_per_node=${NUM_GPUS} \
    --rdzv_endpoint=${MASTER_NODE}:10068 \
    --rdzv_backend=c10d \
    tasks/train_it.py \
    $(dirname $0)/config_7b_stage3.py \
    output_dir ${OUTPUT_DIR}
