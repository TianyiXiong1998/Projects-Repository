import os
import json

class Label():
    def __init__(self):
        pass



    def Load_Data(self, dataset):
        label_dic = {}
        for root,dirs,files in os.walk(dataset):
            for name in files:
                with open(os.path.join(root,name)) as f:
                    if 'White_tailed_Deer' in name:
                        label_dic[name] = 7
        return label_dic

    def Write_in(self,path,dic):
        with open(path,'w', encoding='utf-8') as f:
            for kv in dic.items():
                f.write(kv[0]+' '+ str(kv[1])+'\n')




if __name__ == '__main__':
    label = Label()
    data_dic = label.Load_Data('/s/parsons/h/proj/vision/data/bmgr/cs510bmgr8')

    label.Write_in('/s/chopin/k/grad/xiongty/PA5/White_tailed_Deer.txt',data_dic)
