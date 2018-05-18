import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import datetime
import random

def read_files():
    dataSet = pd.read_csv('flights_train.csv')
    testingSet = pd.read_csv('flights_test.csv')
    return (dataSet, testingSet)

def preprocessData(dataSet):
    #cross validation -- folds, testing 10 and 100
    indices = []
    startIndex = 0
    endIndex = 499
    while endIndex < len(dataSet) - 2:
        indices.append(startIndex)
        indices.append(endIndex)
        startIndex = startIndex + 500
        endIndex = endIndex + 501
        if len(dataSet) - 2 < endIndex:
            indices.append(startIndex)
            indices.append(endIndex)

    #training set : testing set (4:1)
    #training set
    preProcessDataSet = dataSet[indices[0]:indices[3]].copy(deep = True)
    preProcessDataSet = preProcessDataSet.drop(['UNIQUE_CARRIER', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'FIRST_DEP_TIME'], axis = 1)

    return(preProcessDataSet)

def setEncoding(preProcessDataSet):
    #encode origin and destination city
    random.seed()
    airportCodesSet = set()
    while len(airportCodesSet) < len(preProcessDataSet):
        airportCodesSet.add(random.randint(0, len(preProcessDataSet) - 1))

    airportCodesDict = {}
    for index, airport in preProcessDataSet.iterrows():
        if airport['ORIGIN'] in airportCodesDict:
            preProcessDataSet.loc[index, 'ORIGIN'] = airportCodesDict[airport['ORIGIN']]
        else:
            number = random.sample(airportCodesSet, 1)
            airportCodesDict[airport['ORIGIN']] = number[0]
            preProcessDataSet.loc[index, 'ORIGIN'] = airportCodesDict[airport['ORIGIN']]
            airportCodesSet - set(number)

        airportDestination = airport['DEST']
        if airportDestination in airportCodesDict:
            preProcessDataSet.loc[index, 'DEST'] = airportCodesDict[airport['DEST']]
        else:
            number = random.sample(airportCodesSet, 1)
            airportCodesDict[airport['DEST']] = number[0]
            preProcessDataSet.loc[index, 'DEST'] = airportCodesDict[airport['DEST']]
            airportCodesSet - set(number)


    #convert dates to day of year
    for index, row in preProcessDataSet.iterrows():
        date = row['FL_DATE']
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        new_year_day = datetime.datetime(year=date.year, month=1, day=1)
        day = (date - new_year_day).days + 1
        preProcessDataSet.loc[index, 'FL_DATE'] = day
        preProcessDataSet.loc[index, 'DISTANCE'] = int(preProcessDataSet.loc[index, 'DISTANCE'].replace(',', ''))

    return(preProcessDataSet)

#testing scaling
#scaler = StandardScaler()
#preProcessDataSetScaled = pd.DataFrame(scaler.fit_transform(preProcessDataSet))
#preProcessDataSetScaled.to_csv('preprocess_scaled.csv', sep=',', encoding='utf-8')

def runPCA(preProcessDataSet):
    #https://plot.ly/ipython-notebooks/principal-component-analysis/
    droppedLabelsSet = preProcessDataSet.drop(['ARR_DELAY'], axis = 1)
    covarianceMatrix = np.cov(droppedLabelsSet.T)
    eig_vals, eig_vecs = np.linalg.eig(covarianceMatrix)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort()
    eig_pairs.reverse()
    eig_vals.sort()
    print('Eigenvalues were {}'.format(eig_vals))

    #choose k as 4 based on output
    matrixW = np.hstack((eig_pairs[0][1].reshape(15, 1),
                      eig_pairs[1][1].reshape(15, 1),
                      eig_pairs[2][1].reshape(15, 1),
                      eig_pairs[3][1].reshape(15, 1)))

    PCAextractedMatrix = droppedLabelsSet.dot(matrixW)
    PCAextractedMatrixUIDs = PCAextractedMatrix.assign(UID = preProcessDataSet['UID'])

    return(PCAextractedMatrixUIDs)

def runPCATestSet(preProcessDataSet):
    #https://plot.ly/ipython-notebooks/principal-component-analysis/
    covarianceMatrix = np.cov(preProcessDataSet.T)
    eig_vals, eig_vecs = np.linalg.eig(covarianceMatrix)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort()
    eig_pairs.reverse()
    eig_vals.sort()
    print('Eigenvalues were {}'.format(eig_vals))

    #choose k as 4 based on output
    matrixW = np.hstack((eig_pairs[0][1].reshape(15, 1),
                      eig_pairs[1][1].reshape(15, 1),
                      eig_pairs[2][1].reshape(15, 1),
                      eig_pairs[3][1].reshape(15, 1)))

    PCAextractedMatrix = preProcessDataSet.dot(matrixW)
    PCAextractedMatrixUIDs = PCAextractedMatrix.assign(UID = preProcessDataSet['UID'])

    return(PCAextractedMatrixUIDs)

def errorKNN(dataSet, labels):
    #calculate sq. error
    sumOfSquares = 0.0
    for label in labels:
        UID = label[0]
        r = label[1]
        print(label)
        example = dataSet.loc[UID]
        y = example.loc['ARR_DELAY']
        sumOfSquares += (r - y)**2
    return (sumOfSquares / len(labels))

def errorMLP(validationSet, r):
    sumOfSquares = 0.0
    for yValue, rValue in zip(validationSet['ARR_DELAY'], r):
        sumOfSquares += (rValue - yValue)**2

    return (sumOfSquares / len(validationSet))

def writeFile(labels, method):
    cLabels = ['Ids', 'Delay']
    list = []
    for UID, Delay in labels:
        list.append([UID, Delay])
    dfLabels = pd.DataFrame.from_records(list, columns=cLabels)
    filename = method + '_output.csv'
    print(dfLabels)
    dfLabels.to_csv(filename, sep=',')

def kNN(preProcessDataSet, k):
    #k NEAREST NEIGHBOR
    #utilizing simple distance between attributes
    predictions = []
    for curIndex, curRow in preProcessDataSet.iterrows():
        closestNeighbors = [(-1, float('inf'), 0)] * k
        a1 = curRow[0]
        a2 = curRow[1]
        a3 = curRow[2]
        a4 = curRow[3]
        for nIndex, sRow in preProcessDataSet.iterrows():
            if nIndex != curIndex:
                distance = abs(a1 - sRow[0]) + abs(a2 - sRow[1]) + abs(a3 - sRow[2]) + abs(a4 - sRow[3])
                for i in range(0, k):
                    flag = True
                    d = np.array(distance)
                    maxD = np.array(closestNeighbors[i][1]).max()
                    id1 = np.array(sRow['UID'])
                    for item in closestNeighbors:
                        id2 = np.array(item[0])
                        if np.equal(id1, id2):
                            flag = False
                    if np.less(d, maxD) and flag:
                        closestNeighbors[i] = [preProcessDataSet.loc[[nIndex], ['UID']], distance, preProcessDataSet.loc[[nIndex], ['ARR_DELAY']]]
        predictions.append([preProcessDataSet.loc[[curIndex], ['UID']], closestNeighbors])

    labels = []
    for prediction in predictions:
        total_time = 0.0
        for neighbor in prediction[1]:
            attr3 = neighbor[2]
            ARR_DELAY = pd.to_numeric(attr3.loc[attr3.index[0], 'ARR_DELAY'])
            total_time += ARR_DELAY
        avg_time = total_time / k
        UIDobj = prediction[0]
        UID = UIDobj.loc[UIDobj.index[0], ['UID']]
        simpleUID = UID.get('UID')
        labels.append([simpleUID, avg_time])
    return(labels)

def kNNtestSet(testSet, preProcessDataSet, k):
    #k NEAREST NEIGHBOR
    #utilizing simple distance between attributes
    predictions = []
    for curIndex, curRow in testSet.iterrows():
        closestNeighbors = [(-1, float('inf'), 0)] * k
        a1 = curRow[0]
        a2 = curRow[1]
        a3 = curRow[2]
        a4 = curRow[3]
        for nIndex, sRow in preProcessDataSet.iterrows():
            if nIndex != curIndex:
                distance = abs(a1 - sRow[0]) + abs(a2 - sRow[1]) + abs(a3 - sRow[2]) + abs(a4 - sRow[3])
                for i in range(0, k):
                    flag = True
                    d = np.array(distance)
                    maxD = np.array(closestNeighbors[i][1]).max()
                    id1 = np.array(sRow['UID'])
                    for item in closestNeighbors:
                        id2 = np.array(item[0])
                        if np.equal(id1, id2):
                            flag = False
                    if np.less(d, maxD) and flag:
                        closestNeighbors[i] = [preProcessDataSet.loc[[nIndex], ['UID']], distance, preProcessDataSet.loc[[nIndex], ['ARR_DELAY']]]
        predictions.append([testSet.loc[[curIndex], ['UID']], closestNeighbors])

    print('predictions are size {}'.format(len(predictions)))
    print(predictions)

    labels = []
    for prediction in predictions:
        total_time = 0.0
        for neighbor in prediction[1]:
            attr3 = neighbor[2]
            ARR_DELAY = pd.to_numeric(attr3.loc[attr3.index[0], 'ARR_DELAY'])
            total_time += ARR_DELAY
        avg_time = total_time / k
        UIDobj = prediction[0]
        UID = UIDobj.loc[UIDobj.index[0], ['UID']]
        simpleUID = UID.get('UID')
        labels.append([simpleUID, avg_time])

    return(labels)

def MLP(trainingSet, validationSet, train):
    if train:
        #activation - tanh, relu, adam
        #
        mlpReg = MLPRegressor(hidden_layer_sizes=(30, 50, 1 ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', \
        learning_rate = 'adaptive', learning_rate_init=0.05, shuffle=True, tol=0.0001, verbose=False, warm_start=False, \
        early_stopping=False, epsilon=1e-08)
        mlpReg.fit(trainingSet, trainingSet['ARR_DELAY'])
        predictions = mlpReg.predict(validationSet)

    return(predictions)

def MLPtest(testSet, validationSet, train):
    if train:
        #activation - tanh, relu, adam
        #
        mlpReg = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', \
        learning_rate = 'adaptive', learning_rate_init=0.05, shuffle=True, tol=0.1, verbose=False, warm_start=False, \
        early_stopping=False, epsilon=1e-08)
        noLabelsValidationSet = validationSet.drop(['ARR_DELAY'], axis = 1)
        mlpReg.fit(noLabelsValidationSet, validationSet['ARR_DELAY'])
        predictions = mlpReg.predict(testSet)
        labels = []
        for UID, prediction in zip(testSet.loc[:, 'UID'], predictions):
            print(UID, prediction)
            labels.append([UID, float(prediction)])
    print(len(labels))
    return(labels)


##MAIN##
dataSet, testingSet = read_files()
validationSet = dataSet[:1000].drop(['UNIQUE_CARRIER', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'FIRST_DEP_TIME'], axis = 1)
trainingSet = dataSet[1000:].drop(['UNIQUE_CARRIER', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'FIRST_DEP_TIME'], axis = 1)
reducedTestSet = testingSet.drop(['UNIQUE_CARRIER', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'FIRST_DEP_TIME'], axis = 1)
encodedTrainSet = setEncoding(trainingSet)
encodedTestSetEncode = setEncoding(reducedTestSet)
encodedValidationSet = setEncoding(validationSet)
PCAextractedMatrixUIDs = runPCA(encodedTrainSet)
PCAwUIDTest = runPCATestSet(encodedTestSetEncode)
#uncomment kNN to run and predict...beware, it is very lazy!
#k = 13
#kNNlabels = kNNtestSet(PCAwUIDTest, k)
#kNNerror = KNNerror(encodedValidationSet, labels)
#set to train false for prediction mode
MLPlabels = MLPtest(encodedTestSetEncode, encodedValidationSet, train = True)
#errorMLP(validationSet, MLPlabels)

writeFile(MLPlabels, 'MLP')

print('finished....')
##END MAIN##