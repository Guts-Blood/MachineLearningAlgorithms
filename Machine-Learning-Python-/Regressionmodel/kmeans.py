from copy import deepcopy
import numpy as np
import xlrd;
from matplotlib import pyplot as plt

#Define a function to calculate the distance (norm) of two points
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax);
#This is the function to do the k-means clustering
def k_means(Data,k,iteration_number):
    # Generate  random positions for centrals x axes
    C_x = np.random.randint(np.min(Data[:,0]), np.max(Data[:,0]), size=k)
    # generate  random positions for centrals y axes
    C_y = np.random.randint(np.min(Data[:,1]), np.max(Data[:,1]), size=k)
    C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    #Show the original data and original random centrals
    plt.scatter(Data[:,0], Data[:,1], c='black', s=6)
    plt.scatter(C_x, C_y, c='red', s=6)
    plt.show()
    # central points before iteration
    C_old = np.zeros(C.shape)
    print(C)
    # give a var to store the label for each points
    clusters = np.zeros(len(Data))
    # 迭代标识位，通过计算新旧中心点的距离
    iteration_flag = dist(C, C_old, 1)
    #iteration index
    tmp = 1
    #Define J as distortion value;
    J = np.zeros(iteration_number);
    # If the central point do not change or it dose not loop more than iteration number
    #Continue this loop
    while iteration_flag.any() != 0 and tmp < iteration_number:
        # Calculate the clustering labels for each point

        for i in range(len(Data)):
            # Calculate the distance between each point and central point
            distances = dist(Data[i], C, 1);
            # labels represent the label of closest central to each point
            labels = np.argmin(distances);
            J[tmp-1]=J[tmp-1]+distances.min();
            # Store them in clusters which contains cluster labels for each point
            clusters[i] = labels;
        #Store the present centrals
        C_old = deepcopy(C)
        for i in range(k):
            points = [Data[j] for j in range(len(Data)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)

        # Calculate the distance between old C and new C
        print('Iteration %d' % tmp)
        tmp = tmp + 1
        iteration_flag = dist(C, C_old, 1)
        print("The distance between old centrals and new centrals：", iteration_flag)
    # Show the result in graph
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    # Using different color to represent the
    for i in range(k):
            points = np.array([Data[j] for j in range(len(Data)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='black');
    plt.show();
    #Plot J as a function of iteration times
    iter=[];
    for i in range(1,tmp):
        iter.append(i);
    plt.scatter(iter, J[0:tmp-1], marker='*', s=10, c='black');
    my_x_ticks = np.arange(0, tmp+1, 1)
    plt.xticks(my_x_ticks);
    plt.title('J at each iteration')
    plt.xlabel("iteration number")
    plt.ylabel("J (distortion function)")
    plt.show();
def main():
    #First we read the data from excel
    book = xlrd.open_workbook('D:/Work/Homework 4/Houses.xlsx');
    sheet = book.sheet_by_name('Sheet1');
    #Then we transfer our raw data into required format
    Data = [];
    for i in range(2, sheet.nrows):
        Data.append(sheet.row_values(i));
    Data = np.array(Data);

    k_means(Data,3,20);
if __name__ == '__main__':
    from copy import deepcopy
    import numpy as np
    import xlrd;
    from matplotlib import pyplot as plt
    main() ;
