f1 = open("/home/zy/tvm/tutorials/experiment_res/0812_trace.txt","r")
# f2 = open("/home/zy/tvm/tutorials/experiment_res/byoc_trace.txt","r")

# # 读取两个txt文件
# txt1 = f1.read()
# txt2 = f2.read()
# 按行的方式读取txt文件
txt1 = f1.readlines()
# txt2 = f2.readline()

# # 释放两个文件进程
# f1.close()
# f2.close()

# # 将两个文件中内容按空格分隔开
# line1 = txt1.split()
# line2 = txt2.split()

# 以读取方式打开 diff.txt 文件
outfile = open("/home/zy/tvm/tutorials/experiment_res/processed_0812_trace.txt", "w")

# 循环遍历1号文件中的元素

for line in txt1:
	if "importlib" not in line and "packed_func.py" not in line and "task.py" not in line and "object.py" not in line and "abc.py" not in line and \
    "decoder.py" not in line and "record.py" not in line and "space.py" not in line and "fromnumeric.py" not in line and \
        "_methods.py" not in line and "__init__" not in line and "---" not in line \
        and "base.py" not in line and "registry.py" not in line and "pathlib.py" not in line \
            and "hashlib.py" not in line and "_endian.py" not in line and "platform.py" not in line \
                and "re.py" not in line and "sre_parse.py" not in line:

		outfile.write(line)
    
# outfile.write("Above content in 1. But not in 2.")
# for j in line2:
# 	# 查看2中文件是否在1中存在
# 	if j not in line1:
# 		outfile.write(j)
# outfile.write("Above content in 2. But not in 1.")
print("done")