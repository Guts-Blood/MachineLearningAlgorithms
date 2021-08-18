#Self Organizing Network
import numpy as np
import math as m
import matplotlib as plt
import matplotlib.pyplot as pyplt

import tensorflow as tf;


def euc_dist(v1, v2):
  return np.linalg.norm(v1 - v2)

def manhattan_dist(r1, c1, r2, c2):
  return np.abs(r1-r2) + np.abs(c1-c2)
def closest_node(single_data, map):
  # (row,col) of map node closest to data[t]
  m_rows=len(map)
  m_cols=len(map[0])
  result = [0,0]
  #initialize the closest distance as a big number
  closest_dist = 1.0e10
  for i in range(m_rows):
    for j in range(m_cols):
      cur_dist = euc_dist(map[i][j], single_data)
      #update if the distance of current node is smaller than current closest distance
      if cur_dist < closest_dist:
        closest_dist = cur_dist
        result = [i,j]
  return result

def main():
    # Step1: Load the data from website
    # Input data get from website
    #Maroon, dark read brown red
    #light coral, dark salmon
    Input_data = [[128,0,0],
                    [139,0,0],
                    [165,42,42],
                    [178,34,34],
                    [220,20,60],
                    [255,0,0],
                    [0,128,0],
                    [34,139,34],
                    [0,255,0],
                    [50,205,50],
                    [144,238,144],
                    [152,251,152],
                    [135,206,235],
                    [135,206,250],
                    [25,25,112],
                    [0,0,128],
                    [0,0,139],
                    [0,0,205],
                    [0,0,255],
                    [169,169,169],
                    [192,192,192],
                    [211,211,211],
                    [220,220,220],
                    [245,245,245],
                  ]
    Input_data=np.array(Input_data)/255
    #Step2: Initialize the size of the grid and the dimension for each vecotr in grid
    dimension = 3
    row = 100
    col = 100
    # RangeMax = Rows + Cols
    alpha0 = 0.8
    max_epoch_T = 1000
    sigma_list=[1,10,30,50,70]

    # 2. construct the SOM and do the iteration for each sigma and epoach
    # To speed up the algorithm I calculate a range of negiborhood and set it as 5*5 mat with the winning neuron
    # on the center 2,2
    # dist_map = np.zeros((5, 5, 1))
    # for i in range(5):
    #     for j in range(5):
    #         dist_map[i, j] = m.sqrt(pow((i - 2), 2) + pow((j - 2), 2))

    show_epoach = [20 - 1, 40 - 1, 100 - 1, 1000 - 1]
    for sigma0 in sigma_list:
        map = np.random.random_sample(size=(row, col, dimension))
        pyplt.imshow(map, cmap='gray')
        pyplt.title('initial map')
        pyplt.show()
        print("Initialize weights for a 100*100 SOM")
        # max_epoch_T
        for k in range(1000):
            for t in range(len(Input_data)):
                # if s % (max_epoch_T / 10) == 0: print("step = ", str(s))
                # Calculate alpha
                alpha_k = alpha0 * m.exp(-k / max_epoch_T)
                # Calculate winning neuron
                [closet_row, closet_col] = closest_node(Input_data[t], map)
                # Calculate neighborhood function
                dist_map = np.zeros((row, col, 1))
                for i in range(row):
                    for j in range(col):
                        dist_map[i, j] = m.sqrt(pow((i - closet_row), 2) + pow((j - closet_col), 2))
                sigma_k = sigma0 * m.exp(-k / max_epoch_T)
                h_ij_k = np.exp(-dist_map / (2 * pow(sigma_k,2)))
                map = map + alpha_k * h_ij_k * (Input_data[t] - map)
                pyplt.imshow(map, cmap='gray')
                pyplt.title('sigma=%d with epoch %d' % (sigma0, k + 1))
                pyplt.show()
            if k in show_epoach:
                print(alpha_k * h_ij_k.max())
                print('alpha k is %.4f'%alpha_k)
                print('sigma k is %.4f'%sigma_k)
                # Normalize and show
                pyplt.imshow(map, cmap='gray')
                pyplt.title('sigma=%d with epoch %d' % (sigma0, k + 1))
                pyplt.show()

        print("SON construction complete \n")

if __name__ == '__main__':
    main()


