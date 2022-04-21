#!/bin/bash -x

record_root="/home2/zhangya9/tvm/tutorials/bench_openvino"
target='llvm -mcpu=cascadelake -model=platinum-8280'
dtype='float32'

# "MobileNet-v2-1.0"
# "resnet50-v1"
# "resnet50-v2"
# "squeezenet1.0"
# "squeezenet1.1"
# "vgg16"
# "vgg16-bn"
# "densenet121"
# "inception_v3"
# "shufflenet_v2"
# "efficientnet-b0-pytorch"
# "resnext50_32x4d"
# "wide_resnet50_2"
# "resnest50"

network_list=('resnest50')   # 'anti-spoof-mn3')
cores_list=('4') #('28' '28' '4')    # multi-instances should be the last one
batch_list=('1') #('1' '128' '1')

repeat=100
physical_cores=28

for i in $(seq 1 ${#cores_list[@]}); do
    num_cores=${cores_list[$i-1]}
    batch_size=${batch_list[$i-1]}
    num_groups=$[${physical_cores}/${num_cores}]
    for n in $(seq 1 ${#network_list[@]}); do
        network=${network_list[$n-1]}
        log_root="${record_root}/${network}"
        echo "saving logs to ${log_root}"
        mkdir -p ${log_root}

        export OMP_NUM_THREADS=${num_cores}
        export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
        for j in $(seq 0 $[${num_groups}-1]); do
            start_core=$[${j}*${num_cores}]
            end_core=$[$[${j}+1]*${num_cores}-1]
            benchmark_log="${log_root}/${num_cores}cores_bs${batch_size}_cores${start_core}-${end_core}.log"
            # benchmark_log="${log_root}/dnnl_verbose.log"
            printf "=%.0s" {1..100}; echo
            echo "benchmarking autoscheduler using ${network} with ${num_cores} cores and batchsize=${batch_size} on cores: ${start_core}-${end_core}"
            echo "saving logs to ${benchmark_log}"; echo
            
            if [ ${num_groups} == 1 ]
            then
                numactl --physcpubind=${start_core}-${end_core} --membind=0 \
                python benchmark_byoc.py \
                    --network=${network} \
                    --batch_size=${batch_size} \
                    --target="${target}" \
                    --dtype=${dtype} \
                    --warmup=20 \
                    --batches=${repeat} \
                    | tee ${benchmark_log} 2>&1
                echo "done benchmarking autoscheduler using ${network} with ${num_cores} cores and batchsize=${batch_size} on cores: ${start_core}-${end_core}"
            else
                numactl --physcpubind=${start_core}-${end_core} --membind=0 \
                python benchmark_byoc.py \
                    --network=${network} \
                    --batch_size=${batch_size} \
                    --target="${target}" \
                    --dtype=${dtype} \
                    --warmup=20 \
                    --batches=${repeat} \
                    | tee ${benchmark_log} 2>&1 &
            fi
        done
    done
done
echo "benchmarking sessions lanched, please wait for the python runs."
