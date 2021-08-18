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
    # 可视化划分结果，创建一个颜色的list
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '0.1', '0.4', '0.8', '0.5']
    '''
    DBSCAN method 使用DBscan来进行聚类，DBscan是基于密度来划分，比如eps=0.05 min population=1000
    表示仅有半径为0.05区域且数据点数量大于1000的区域会被局类和划分为一个族群，类似于寻找大密度的指定面积区域
    '''
    from sklearn.cluster import DBSCAN
    #Due to the size of data I must delete some data to make the Algorithm run
    #因为数据总数太大，所以随机删除一些数据点，删除方式选择的是打乱顺序，然后截取前100000个数据点来运用
    seed=50
    np.random.seed(seed)
    np.random.shuffle(data_normalize)
    np.random.seed(seed)
    np.random.shuffle(data)
    '''#eps=0.05 and min popularion=1000'''
    eps_value=0.05
    min_pop=1000
    clustering = DBSCAN(eps=eps_value,min_samples=min_pop,metric='manhattan',algorithm='kd_tree').fit(data_normalize[0:100000,1:3])
    label_db=clustering.labels_

    n_clusters_ = len(set(label_db)) - (1 if -1 in label_db else 0)
    n_noise_ = list(label_db).count(-1)
    print('DBscan has %d cluster'%n_clusters_)
    print('DBscan has %d noise points'%n_noise_)
    #截取前100000个点来作为接下来使用的数据来进行作图
    data_trun=data[0:100000]
    # for i in range(num_cluster):
    #     plt.scatter(data[label_db==i,2],data[label_db==i,1],color=color_list[i],label='cluster'+str(i+1))
    #     plt.legend()
    # plt.show()

    unique_labels = set(label_db)

    for k in unique_labels:
        if k == -1:
            # Black used for noise.
            plt.scatter(data_trun[label_db == k, 2], data_trun[label_db == k, 1], color='black')
        else:
            plt.scatter(data_trun[label_db==k,2],data_trun[label_db==k,1],color=color_list[k+1])

    plt.title('Dbscan with eps=%.4f and min population=%d' % (eps_value,min_pop))
    plt.show()



    '''#eps=0.05 and min popularion=1000'''
    eps_value=0.05
    min_pop=40000
    clustering = DBSCAN(eps=eps_value,min_samples=min_pop,metric='manhattan',algorithm='kd_tree').fit(data_normalize[0:100000,1:3])

    label_db=clustering.labels_

    n_clusters_ = len(set(label_db)) - (1 if -1 in label_db else 0)
    n_noise_ = list(label_db).count(-1)
    print('DBscan has %d cluster'%n_clusters_)
    print('DBscan has %d noise points'%n_noise_)

    data_trun=data[0:100000]
    # for i in range(num_cluster):
    #     plt.scatter(data[label_db==i,2],data[label_db==i,1],color=color_list[i],label='cluster'+str(i+1))
    #     plt.legend()
    # plt.show()

    unique_labels = set(label_db)

    for k in unique_labels:
        if k == -1:
            # Black used for noise.
            plt.scatter(data_trun[label_db == k, 2], data_trun[label_db == k, 1], color='black')
        else:
            plt.scatter(data_trun[label_db==k,2],data_trun[label_db==k,1],color=color_list[k+1])


    plt.title('Dbscan with eps=%.4f and min population=%d' % (eps_value,min_pop))
    plt.show()

    #Show how many people in each cluster
    num_people=[]
    for i in range(num_cluster):
        num_people.append(sum(labels_area==i))

    '''#eps=0.03 and min popularion=10000'''
    eps_value=0.03
    min_pop=10000
    clustering = DBSCAN(eps=eps_value,min_samples=min_pop,metric='manhattan',algorithm='kd_tree').fit(data_normalize[0:100000,1:3])

    label_db=clustering.labels_

    n_clusters_ = len(set(label_db)) - (1 if -1 in label_db else 0)
    n_noise_ = list(label_db).count(-1)
    print('DBscan has %d cluster'%n_clusters_)
    print('DBscan has %d noise points'%n_noise_)

    data_trun=data[0:100000]
    # for i in range(num_cluster):
    #     plt.scatter(data[label_db==i,2],data[label_db==i,1],color=color_list[i],label='cluster'+str(i+1))
    #     plt.legend()
    # plt.show()
    unique_labels = set(label_db)

    for k in unique_labels:
        if k == -1:
            # Black used for noise.
            plt.scatter(data_trun[label_db == k, 2], data_trun[label_db == k, 1], color='black')
        else:
            plt.scatter(data_trun[label_db==k,2],data_trun[label_db==k,1],color=color_list[k+1])

    plt.title('Dbscan with eps=%.4f and min population=%d' % (eps_value,min_pop))
    plt.show()

    '''#eps=0.01 and min popularion=10000，缩小区域'''
    eps_value=0.02
    min_pop=300
    clustering = DBSCAN(eps=eps_value,min_samples=min_pop,metric='manhattan',algorithm='kd_tree').fit(data_normalize[0:100000,1:3])

    label_db=clustering.labels_

    n_clusters_ = len(set(label_db)) - (1 if -1 in label_db else 0)
    n_noise_ = list(label_db).count(-1)
    print('DBscan has %d cluster'%n_clusters_)
    print('DBscan has %d noise points'%n_noise_)

    data_trun=data[0:100000]
    # for i in range(num_cluster):
    #     plt.scatter(data[label_db==i,2],data[label_db==i,1],color=color_list[i],label='cluster'+str(i+1))
    #     plt.legend()
    # plt.show()

    unique_labels = set(label_db)

    for k in unique_labels:
        if k == -1:
            # Black used for noise.
            plt.scatter(data_trun[label_db == k, 2], data_trun[label_db == k, 1], color='black')
        else:
            plt.scatter(data_trun[label_db==k,2],data_trun[label_db==k,1],color=color_list[k+1])

    plt.title('Dbscan with eps=%.4f and min population=%d' % (eps_value,min_pop))
    plt.show()

