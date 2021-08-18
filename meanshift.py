if __name__ == '__main__':
    import numpy as np;
    import pandas as pd;
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from sklearn.cluster import MeanShift,estimate_bandwidth
    '''
    Data preprocessing:数据读取和处理
    从csv中读取数据然后转化成numpy的类型方便后续的处理
    '''
    data = pd.read_csv('/Users/jiaweiqian/Desktop/Work/KmeansNLP/San_Francisco_154k.csv')
    data=data.to_numpy()
    #ignore the date and year: only focus on the time period
    #对于时间数据去掉年份和日期，只保留小时和分钟,后续筛选了发现没有很大区分度
    for i in range(len(data[:,0])):
        str_l=len(data[i,0])
        data[i,0]=int(data[i,0][str_l-5:str_l-3])+int(data[i,0][str_l-2:])/60
    #Normalize all features to [0,1] 归一化所有数据到0，1 的区间
    data_normalize=data.copy()
    data_normalize[:,0]=(data_normalize[:,0]-data_normalize[:,0].min())/(data_normalize[:,0].max()-data_normalize[:,0].min())
    data_normalize[:,1]=(data_normalize[:,1]-data_normalize[:,1].min())/(data_normalize[:,1].max()-data_normalize[:,1].min())
    data_normalize[:,2]=(data_normalize[:,2]-data_normalize[:,2].min())/(data_normalize[:,2].max()-data_normalize[:,2].min())
    q=0.2
    min_f=2000
    bandwidth = estimate_bandwidth(data_normalize[:,1:3], quantile=q, n_samples=10000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True,min_bin_freq=min_f)
    ms.fit(data_normalize[:, 1:3])
    color_list=['black','green','red','yellow','sienna','snow','blue','purple','grey','darkred','tomato','chocolate','moccasin','gold','lime','greenyellow','olivedrab','hotpink','pink','navy']
    #对于每个颜色划分
    centers=ms.cluster_centers_
    for i in range(len(centers)):
        plt.scatter(data[ms.labels_==i,2],data[ms.labels_==i,1],color=color_list[i%20],label='cluster'+str(i+1))
        # plt.legend()
    plt.title("Quantile=%.2f,min_bin_freq=%d" %(q,min_f))
    plt.show()