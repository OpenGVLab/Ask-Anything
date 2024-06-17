export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
export OUTPUT_DIR='.'
echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=1
MASTER_NODE='SH-IDC1-10-140-1-1'

torchrun  --nnodes=${NNODE} --nproc_per_node=${NUM_GPUS} \
    tasks/train_pt.py \
    $(dirname $0)/config_7b_stage2.py \
    output_dir ${OUTPUT_DIR}
