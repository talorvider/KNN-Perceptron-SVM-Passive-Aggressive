import numpy as np
import sys
import os

train_x_path, train_y_path, tests_x_path, output_log_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]


def zScore(trainData, mean, stand_dev):
    
    #calculate new values for each column
    fituresLen = len(trainData[0])
    for i in range(fituresLen):
        trainData[:,i] -=mean[i]
        if (stand_dev[i]!=0.0):
            trainData[:,i] /=stand_dev[i]
    return trainData

def calcDist(vec1, vec2):
    distance = 0.0
    fituresLen = len(vec1)
    for i in range(fituresLen):
        distance += (vec1[i] - vec2[i])**2
    return np.sqrt(distance)

def findKNeighbords(trainData, checkRow, Kneighbors):
    closestNeighborsIndx = []
    allDistan = []
    for i , trainRow in enumerate(trainData):
        dist = calcDist(trainRow, checkRow)
        allDistan.append((i, dist))
    #find the k nearst neighbors
    allDistan.sort(key=lambda x: x[1])
    for j in range(Kneighbors):
        closestNeighborsIndx.append(allDistan[j][0])
    return closestNeighborsIndx

def predictClass(closestNeighborsIndx, trainLabel):
    labels = [trainLabel[i] for i in closestNeighborsIndx]
    prediction = max(set(labels), key=labels.count)
    return prediction

def KNN():
    trainData = np.loadtxt(train_x_path,delimiter =",")
    trainLabelSet = np.loadtxt(train_y_path,delimiter =",")
    testData = np.loadtxt(tests_x_path,delimiter =",")
    #normal values by z-score
    #get mean and stand_dev for each column on train data 
    train_mean = np.mean(trainData, axis=0)
    train_stand_dev = np.std(trainData, axis=0)
    trainDataSet = zScore(trainData, train_mean, train_stand_dev)
    #normlize test data set by train values
    normalizeTestData = zScore(testData, train_mean, train_stand_dev)
    k = 5  
    predictions = []
    for testRow in normalizeTestData:
        closestNeighborsIndx = findKNeighbords(trainDataSet, testRow, k)
        prediction = predictClass(closestNeighborsIndx, trainLabelSet)
        predictions.append(int(prediction))
    return predictions

def Preceptron():
    trainData = np.loadtxt(train_x_path,delimiter =",")
    trainLabelSet = np.loadtxt(train_y_path,delimiter =",")
    testData = np.loadtxt(tests_x_path,delimiter =",")
    #normal values by z-score
    #get mean and stand_dev for each column on train data (90%)
    train_mean = np.mean(trainData, axis=0)
    train_stand_dev = np.std(trainData, axis=0)
    trainDataSet = zScore(trainData, train_mean, train_stand_dev)
    #normlize test data set by train values
    normalizeTestData = zScore(testData, train_mean, train_stand_dev)
    #initilaize array with wiegth for each label, last column is for bias
    w = np.zeros((3, len(trainDataSet[0]) + 1))
    # add 1 to evry fitcure in train set for multiply in bias
    one_array = np.ones((len(trainDataSet), 1))
    trainDataSet = np.append(trainDataSet, one_array, axis=1)
     # add 1 to evry fitcure in test set for multiply in bias after normalization
    one_array = np.ones((len(normalizeTestData), 1))
    normalizeTestData = np.append(normalizeTestData, one_array, axis=1)
 #learning w from train set
    for e in range(250):
        shuffler = np.random.permutation(len(trainDataSet))
        trainDataSet = trainDataSet[shuffler]
        trainLabelSet = trainLabelSet[shuffler]
        for x, y in zip(trainDataSet, trainLabelSet):
            y = int(y)
            #predict which of the weight vectors (w) gave the max value
            #add 1 for evry point in data for the bias
            yTag = np.argmax(np.dot(w,x))
            #if the predict is wrong change the weight vector
            if (yTag != y):
                #add more weight for the correct vector label
                w[y,:] = np.add(w[y,:] ,(0.1 * x))
                #Subtraction weight for the correct vector label
                w[yTag,:] = w[yTag,:] - 0.1 * x 

     #run on test set 
    predictions = []
    for x in normalizeTestData:
        prediction = np.argmax(np.dot(w,x))
        predictions.append(prediction)
    return predictions

def SVM():
    trainData = np.loadtxt(train_x_path,delimiter =",")
    trainLabelSet = np.loadtxt(train_y_path,delimiter =",")
    testData = np.loadtxt(tests_x_path,delimiter =",")
    #normal values by z-score
    #get mean and stand_dev for each column on train data (90%)
    train_mean = np.mean(trainData, axis=0)
    train_stand_dev = np.std(trainData, axis=0)
    trainDataSet = zScore(trainData, train_mean, train_stand_dev)
    #normlize test data set by train values
    normalizeTestData = zScore(testData, train_mean, train_stand_dev)
    #initilaize array with wiegth for each label, last column is for bias
    w = np.zeros((3, len(trainDataSet[0]) + 1)) 
    # add 1 to evry fitcure in train set for multiply in bias
    one_array = np.ones((len(trainDataSet), 1))
    trainDataSet = np.append(trainDataSet, one_array, axis=1)
        # add 1 to evry fitcure in test set for multiply in bias after normalization
    one_array = np.ones((len(normalizeTestData), 1))
    normalizeTestData = np.append(normalizeTestData, one_array, axis=1)
    #start update w by evry x in train set 
    
    for e in range(40):
        shuffler = np.random.permutation(len(trainDataSet))
        trainDataSet = trainDataSet[shuffler]
        trainLabelSet = trainLabelSet[shuffler]
        for x, y in zip(trainDataSet, trainLabelSet):
            y = int(y)
            #get the higth wi from w when i !=y
            allWVal =[]
            for i in range(3):
                if (i != y):
                    yTagVal = np.dot(w[i],x)
                    allWVal.append((yTagVal, i))
            yTag = max(allWVal, key=lambda item:item[0])[1]
            #allredy add 1 for evry point in data for the bias
            #update rule for w[y] and w[yTag]
            calcWdist = np.dot(w[y],x) - np.dot(w[yTag],x)
            lossVal = max([0, 1- calcWdist])
            if (lossVal >0):
                t = 0.01 * 0.1
                calc = 1 - t
                w[y] = np.add(np.dot(w[y],calc) , np.dot(0.1,x))
                w[yTag] = np.subtract(np.dot(w[yTag],calc) , np.dot(0.1,x))
                for j in range(3):
                    if (j != y and j!=yTag):
                        w[i] = np.dot(w[i],calc)
            else:
                for j in range(3):
                        if (j != y and j!=yTag):
                            w[i] = np.dot(w[i],calc)
        
    #run on test set 
    predictions = []
    for x in normalizeTestData:
        prediction = np.argmax(np.dot(w,x))
        predictions.append(prediction)
    return predictions   

def passiveAggressive():
    trainData = np.loadtxt(train_x_path,delimiter =",")
    trainLabelSet = np.loadtxt(train_y_path,delimiter =",")
    testData = np.loadtxt(tests_x_path,delimiter =",")
    #normal values by z-score
    #get mean and stand_dev for each column on train data (90%)
    train_mean = np.mean(trainData, axis=0)
    train_stand_dev = np.std(trainData, axis=0)
    trainDataSet = zScore(trainData, train_mean, train_stand_dev) 
    #normlize test data set by train values
    normalizeTestData = zScore(testData, train_mean, train_stand_dev) 
    #initilaize array with wiegth for each label, last column is for bias
    w = np.zeros((3, len(trainDataSet[0]) + 1))
    # add 1 to evry fitcure in train set for multiply in bias
    one_array = np.ones((len(trainDataSet), 1))
    trainDataSet = np.append(trainDataSet, one_array, axis=1)
     # add 1 to evry fitcure in test set for multiply in bias after normalization
    one_array = np.ones((len(normalizeTestData), 1))
    normalizeTestData = np.append(normalizeTestData, one_array, axis=1)
    #start update w by evry x in train set 
    
    for e in range(19):
        shuffler = np.random.permutation(len(trainDataSet))
        trainDataSet = trainDataSet[shuffler]
        trainLabelSet = trainLabelSet[shuffler]
        for x, y in zip(trainDataSet, trainLabelSet):
            y = int(y)
            #get the higth wi from w when i !=y
            allWVal =[]
            for i in range(3):
                if (i != y):
                    yTagVal = np.dot(w[i],x)
                    allWVal.append((yTagVal, i))
            yTag = max(allWVal, key=lambda item:item[0])[1]
            #allredy add 1 for evry point in data for the bias
            #update rule, wite tuae (not hyper parameter)
            calcWdist = np.dot(w[y],x) - np.dot(w[yTag],x)
            lossVal = max([0, (1- calcWdist)])
            normPow2 = np.dot(x,x)
            tuae =  lossVal / (2 * normPow2)
            w[y] = np.add(w[y] , np.dot(tuae,x))
            w[yTag] = np.subtract(w[yTag], np.dot(tuae,x))

    
    #run on test set 
    predictions = []
    for x in normalizeTestData:
        prediction = np.argmax(np.dot(w,x))
        predictions.append(prediction)
    return predictions
            
def main():
    testData = np.loadtxt(tests_x_path,delimiter =",")
    KNNPredict = KNN()
    preceptronPredict= Preceptron() 
    svmPredict = SVM() 
    PaPredict = passiveAggressive()
    outFile = os.open(output_log_path, os.O_RDWR | os.O_CREAT)
    for i in range(len(testData)):
        line = f"knn: {KNNPredict[i]}, perceptron: {preceptronPredict[i]}, svm: {svmPredict[i]}, pa: {PaPredict[i]}\n"
        l = str.encode(line)
        numBytes = os.write(outFile, l)
    os.close(outFile)
    
main()
