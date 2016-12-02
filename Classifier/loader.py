
import os
import numpy as np

'''
return a dictionary of [{vin:,label:}]
Uber - 0
private - 1
'''
def label_loader(folder):
    allSeries = {}
    path = "data/"+folder+"/label.csv"
    with open(path,'r') as f:
        for line in f: #read in record
            tuples = line.split(",")
            key = tuples[0]
            val = 0 if(tuples[1][0]=='U')else 1
            allSeries[key]= val
    return allSeries

'''
input training vin, test vin
output:
    training -- list of dict{"data","label"}
    vin_testing -- dict of {vin:{data:[] of feature, label:label}}
    record_testing -- list of dict{"data","label"}
'''
def CNN_feature_loader(trainVin,testVin,folder):
    holiday = ["2016-4-30","2016-5-1","2016-5-2","2016-5-7","2016-5-8"]
    workday = ["2016-4-29","2016-5-3","2016-5-4","2016-5-5","2016-5-6","2016-5-9","2016-5-10","2016-5-11"]
    labelList = label_loader(folder)
    labelListKeys = label_loader(folder).keys()
    np.random.shuffle(labelListKeys)
    path = "data/"+folder+"/matrix/"
    allFiles = os.listdir(path)
    training = []
    recordTesting = []
    testing = {}.fromkeys(testVin)
    for file in allFiles:
        mat = np.loadtxt(path+file).reshape(1,576)
        splits = file.split(".")
        vin = splits[0]
        date = splits[1]
        lb = int(labelList[vin])
        a = [0,0]
        a[lb] = 1
        label = np.asarray(a).reshape(1,2)
        dict = {"data":mat,"label":label}
        if(vin in trainVin):
            training.append(dict)
            if(date in workday):
                training.append(dict)
                training.append(dict)
        else:
            if(vin in testVin):
                if(testing[vin]==None):
                    testing[vin]={"data":[],"label":label}
                testing[vin]["data"].append(mat)
                if(date in workday):
                    testing[vin]["data"].append(mat)
                    testing[vin]["data"].append(mat)
                recordTesting.append(dict)
                if(date in workday):
                    recordTesting.append(dict)
                    recordTesting.append(dict)
    return {"training":training,"vin_testing":testing,"record_testing":recordTesting}


'''
input training vin and test vin
output  training feature
        training label
        test feature
        test label
'''
def svm_feature_loader(trainVin,testVin,folder):
    allLabel = label_loader(folder)
    tr_featureList = []
    tr_labelList = []
    tst_featureList = {}
    tst_labelList = {}
    path = "data/"+folder+"/feature_extracted.txt"
    with open(path,'r') as f:
        for line in f: #read in record
            tuples = line.split(":")
            vin = tuples[0]
            featurestr = tuples[1].rstrip()
            features = map(lambda x:float(x),featurestr.split(","))
            label = allLabel[vin]

            if(vin in trainVin):
                tr_featureList.append(features)
                tr_labelList.append(label)
            else:
                if(vin in testVin):
                    tst_featureList[vin] =  features
                    tst_labelList[vin] = label
    return {"trainingset":{"featureList":tr_featureList,"labelList":tr_labelList},"testset":{"featureList":tst_featureList,"labelList":tst_labelList}}