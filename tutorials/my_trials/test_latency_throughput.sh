OMP_NUM_THREADS=28 \
KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 \
numactl --physcpubind=0-27 --membind=0 \
python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py \
--check_acc=True >> $HOME/tvm/tutorials/experiment_res/0916/opt_byoc_acc.txt

# OMP_NUM_THREADS=28 \
# KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 \
# numactl --physcpubind=0-27 --membind=0 \
# python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py \
# --batch-size=1

# OMP_NUM_THREADS=28 \
# KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 \
# numactl --physcpubind=0-27 --membind=0 \
# python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py \
# --batch-size=1 --profiling=True >> $HOME/tvm/tutorials/experiment_res/0916/opt_byoc_bs1.txt

# OMP_NUM_THREADS=28 \
# KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 \
# numactl --physcpubind=0-27 --membind=0 \
# python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py \
# --batch-size=128

# OMP_NUM_THREADS=28 \
# KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 \
# numactl --physcpubind=0-27 --membind=0 \
# python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py \
# --batch-size=128 --profiling=True >> $HOME/tvm/tutorials/experiment_res/0916/opt_byoc_bs128.txt