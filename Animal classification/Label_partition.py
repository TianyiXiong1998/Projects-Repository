import os
from tkinter.filedialog import Open

class Label_partition():
    def __init__(self):
        pass



    def Label(self, path):
        label = []
        train_Label = []
        test_Label = []
        with open(path,'r') as f:
            for line in f.readlines():
                label.append(line)
        
        train_Label = label[0:int(0.8*len(label))]
        test_Label = label[int(0.8*len(label)):]
        return train_Label,test_Label
    
        

    def Write_in(self,path,label):
        with open(path,'a', encoding='utf-8') as f:
            for kv in label:
                f.write(kv.split(" ")[0]+' '+ str(kv.split(" ")[1]))

if __name__ == '__main__':
    label = Label_partition()
    train_label,test_label = label.Label('/s/chopin/k/grad/xiongty/PA5/Bighorn_sheep.txt')

    label.Write_in('/s/chopin/k/grad/xiongty/PA5/train_label.txt',train_label)
    label.Write_in('/s/chopin/k/grad/xiongty/PA5/Bighorn_sheep_test_label.txt',test_label)