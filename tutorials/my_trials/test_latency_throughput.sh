#!/bin/bash
export OMP_NUM_THREADS=28
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
# OMP_NUM_THREADS=28 \
# KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 \
# numactl --physcpubind=0-27 --membind=0 \
# python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py --batch-size=1 #--check_acc=False

numactl --physcpubind=0-27 --membind=0 \
python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py --batch-size=128

# for ((i=0; i<7; i++))
#     do
#         startCore=$[${i}*4]
#         endCore=$[${startCore}+3]
#         numactl --physcpubind=${startCore}-${endCore} --membind=0 \
#         python /home2/zhangya9/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py& #--profiling=True& #>> /home2/zhangya9/tvm/tutorials/experiment_res/0917/opt_byoc_${i}_v1.7.txt &
#     done
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
