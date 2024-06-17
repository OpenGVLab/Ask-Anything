srun -p video5 -N1 -n2 --job-name=vllm_vc2 --cpus-per-task=8 --gres=gpu:4 --quotatype=auto python vllm_test.py
