#!/bin/bash
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_PORT=$((10000 + $RANDOM % 100))

echo "All nodes used:"
echo ${ALL_NODES}
echo "Master node:"
echo ${MASTER_NODE}
echo "Args:"
echo $@

torchrun --rdzv_endpoint=${MASTER_NODE}:10066 $@
