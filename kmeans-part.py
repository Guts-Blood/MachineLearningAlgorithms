if __name__ == '__main__':
    import numpy as np;
    import pandas as pd;
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
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


    #设定聚类总数为1~20
    '''使用k-means的算法来进行区域划分，k-means原理是基于数据点到中心点的距离来划分每个数据点的类别'''
    num_cluster=20

    #使用kemans的算法来将数据划分为3~9个区域
    kmeans_area = KMeans(n_clusters=num_cluster, random_state=10)
    kmeans_area.fit(data_normalize[:,1:3])
    labels_area=kmeans_area.labels_
    #可视化划分结果，创建一个颜色的list
    color_list=['black','green','red','yellow','sienna','snow','blue','purple','grey','darkred','tomato','chocolate','moccasin','gold','lime','greenyellow','olivedrab','hotpink','pink','navy']
    #对于每个颜色划分
    for i in range(num_cluster):
        plt.scatter(data[labels_area==i,2],data[labels_area==i,1],color=color_list[i%20],label='cluster'+str(i+1))
        # plt.legend()
    #Show how many people in each cluster 显示每个区域有多少人口
    num_people=[]
    for i in range(num_cluster):
        num_people.append(sum(labels_area==i))
    #还原center到原来的坐标并且print出来
    centers=kmeans_area.cluster_centers_

    centers[:,0]=data[:,1].min()+centers[:,0]*(data[:,1].max()-data[:,1].min())
    centers[:,1]=data[:,2].min()+centers[:,1]*(data[:,2].max()-data[:,2].min())
    plt.scatter(centers[:, 0], centers[:, 1], color='saddlebrown', marker='o', linestyle='dashed')
    plt.title('kemans k=%d' % num_cluster)
    # # xmin, xmax, ymin, ymax = plt.axis([35, 39, -125, -115])
    plt.axis([-122.6,-122.25,37.6,37.9])
    plt.show()
    print('K-means algorithm with k=%d centers are: '%num_cluster)
    print(centers)


    # for order in range(len(centers)):
    #     plt.annotate('Center %d'%order,(centers[order][0],centers[order][1]))


    '''此操作没有太大的区分度和用处'''
    #对于每个地区的twitter使用时间进行分析，将使用时间分成3类并显示每一类的中心
    #Take out the data from the 9 areas and analysis the time the use Twitter
    # for i in range(num_cluster):
    #     kmeans_time = KMeans(n_clusters=3, random_state=0)
    #     kmeans_time.fit(data[labels_area==i,0].reshape(sum(labels_area==i),1))
    #     print('The most likely time of using Twitter in area %d is : '%(i+1))
    #     centers=kmeans_time.cluster_centers_
    #     centers.sort(axis=0)
    #     print(centers)


