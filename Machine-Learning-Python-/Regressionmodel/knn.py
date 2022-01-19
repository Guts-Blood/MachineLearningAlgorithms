import xlrd;
import numpy as np;

def knn(trainData, testData, labels, k):
    # Get the size of training data
    rowSize = trainData.shape[0];
    # Calculate the distance between the test Data and training Data
    diff = np.tile(testData, (rowSize, 1)) - trainData;
    sqrDiff = diff ** 2;
    sqrDiffSum = sqrDiff.sum(axis=1);
    distances = sqrDiffSum ** 0.5;
    # Sort the distance
    sortDistance = distances.argsort()
    #Count is used to store the appeared times of each class
    count = {}
    for i in range(k):
        vote = labels[sortDistance[i]]
        count[vote] = count.get(vote, 0) + 1
    # Find the count with highest appearance value
    max_count=0;
    for j in count:
        if count[j]>max_count:
            predict_label = j;
            max_count=count[j];

    return predict_label;

def main():
    #First we read the data from excel
    book = xlrd.open_workbook('D:/Work/Homework 4/Glasses.xlsx');
    sheet = book.sheet_by_name('Sheet1');
    #Then we transfer our raw data into required format
    Data = [];
    for i in range(sheet.nrows):
        Data.append(sheet.row_values(i));
    #Transfer Data into its features and labels
    Features = [];
    Labels = [];
    for i in range(2, len(Data)):
        Features.append(Data[i][0:6]);
        Labels.append(Data[i][6])
    Features=np.array(Features);
    Labels=np.array(Labels);
    #Then the test Data will be
    Test = [[70, 0, 0, 0, 20, 10], [70, 0, 0, 15, 0, 15], [70, 0, 0, 15, 15, 0]];
    for j in range(len(Test)):
        print(knn(Features, np.array(Test[j]), Labels, 3))
if __name__ == '__main__':
    main();
