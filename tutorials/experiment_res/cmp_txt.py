f1 = open("/home/zy/tvm/tutorials/experiment_res/processed_tvm_trace_0349.txt","r")
f2 = open("/home/zy/tvm/tutorials/experiment_res/processed_byoc_trace_0349.txt","r")

# 读取两个txt文件
l1 = f1.readline()
l2 = f2.readline()
cnt = 0
lst = []
while l2:
    cnt+=1
    if l2!=l1:
        print(cnt)
        lst.append(cnt)
        if len(lst)==60:
            break
    l1 = f1.readline()
    l2 = f2.readline()
print(lst)
print("done")