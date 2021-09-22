#!/bin/bash
export OMP_NUM_THREADS=4
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

### realtime fps
# for ((i=0; i<7; i++))
#     do
#         startCore=$[${i}*4]
#         endCore=$[${startCore}+3]
#         numactl --physcpubind=${startCore}-${endCore} --membind=0 \
#         python /home2/zhangya9/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py \
#         --batch-size=1 & 
#     done


### realtime all ops' execution time
for ((i=0; i<7; i++))
    do
        startCore=$[${i}*4]
        endCore=$[${startCore}+3]
        numactl --physcpubind=${startCore}-${endCore} --membind=0 \
        python /home2/zhangya9/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py& #--profiling=True& #>> /home2/zhangya9/tvm/tutorials/experiment_res/0917/opt_byoc_${i}_v1.7.txt &
    done
