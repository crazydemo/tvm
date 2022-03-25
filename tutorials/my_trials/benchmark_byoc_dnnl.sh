# !/bin/bash
export OMP_NUM_THREADS=28
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

numactl --physcpubind=0-27 --membind=0 \
python $HOME/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py --batch_size=1 # > $HOME/tvm/tutorials/my_trials/0310_tvm_resnet18_v2_json.txt 2>&1
# python $HOME/tvm/tests/python/contrib/test_dnnl.py



# export OMP_NUM_THREADS=4
# export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

# for ((i=0; i<7; i++))
#     do
#         startCore=$[${i}*4]
#         endCore=$[${startCore}+3]
#         numactl --physcpubind=${startCore}-${endCore} --membind=0 \
#         python /home2/zhangya9/tvm/tutorials/my_trials/benchmark_byoc_dnnl.py& # > $HOME/tvm/tutorials/my_trials/0307_onnx_verbose${i}.txt 2>&1 &
#     done
