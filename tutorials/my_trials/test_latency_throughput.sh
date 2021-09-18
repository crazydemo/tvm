#!/bin/bash
export OMP_NUM_THREADS=28
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

### check acc
# numactl --physcpubind=0-27 --membind=0 \
# python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py \
# --check_acc=True

## latency fps
# numactl --physcpubind=0-27 --membind=0 \
# python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py \
# --batch-size=1

### latency all ops' execution time
#numactl --physcpubind=0-27 --membind=0 \
#python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py \
#--batch-size=1 --profiling=True # >> $HOME/tvm/tutorials/experiment_res/0917/opt_byoc_bs1.txt

### throughput fps
# numactl --physcpubind=0-27 --membind=0 \
# python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py \
# --batch-size=128

### throughput all ops' execution time
numactl --physcpubind=0-27 --membind=0 \
python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py \
--batch-size=1 #>> $HOME/tvm/tutorials/experiment_res/0917/opt_byoc_bs128_v1.7.txt
