#import torch
#a=torch.rand(3,2)
#print(a)
 
#a=a[torch.randperm(a.size(0))]
#print(a)

#a=a[:,torch.randperm(a.size(1))]
#print(a)

import numpy as np
import random

out=np.random.rand(2,5,2)#创建多维数组
print(out)
#time_index_list=[n for n in range(len(X[0]))]#提取X第一维度，即30个数的顺序
#print('time_index_list=',time_index_list)
#np.random.shuffle(time_index_list)#随机打乱这30个数的顺序
#print('shuffle time_index_list',time_index_list)
##打乱顺序的方法有两种：
##np.random.shuffle() 和np.random.permutation(),前者改变原有数据、后者不改变原有数据
#save_shuff_x=[]
#for i in time_index_list:
#    shuff_x=X[:,i,:]#按照随机打乱的顺序组合成新的new_x
#    save_shuff_x.append(np.expand_dims(shuff_x,axis=1))
#    print('shuff_x=',shuff_x)
#    print('save_shuff_x=',save_shuff_x)
#new_x=np.concatenate(save_shuff_x,axis=1)
#print(new_x)
#print(new_x.shape)#（30，2，3）

random_noise1 = random.randint(0, out.shape[0] - 1)
out_neg41 = out
out_neg41[random_noise1:random_noise1+1 , : , :] = 0
random_noise2 = random.randint(0, out.shape[1] - 1)
out_neg42 = out
out_neg42[: , random_noise2:random_noise2+1 , :] = 0
#out_neg4 = choice([out_neg41,out_neg42])
#print(out_neg41)
print(out_neg42)
