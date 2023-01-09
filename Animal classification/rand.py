import random
 
out_file = open('/s/chopin/k/grad/xiongty/PA5/train_labels.txt','w',encoding='utf-8')  #输出文件位置
 
lines = []
 
with open('/s/chopin/k/grad/xiongty/PA5/train_label.txt', 'r',encoding='utf-8') as f:   #需要打乱的原文件位置
    for line in f:  
        lines.append(line)
random.shuffle(lines)
 
for line in lines:
    out_file.write(line)
 
