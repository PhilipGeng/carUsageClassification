import loader as loader
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

testDataFolder = "data_test"
trainDataFolder = "data_train"
tstvin = np.asarray(loader.label_loader(testDataFolder).keys())
trvin = np.asarray(loader.label_loader(trainDataFolder).keys())

tstdata = loader.svm_feature_loader([],tstvin,testDataFolder)['testset']
tst_featrure = [tstdata['featureList'][vin] for vin in tstvin]
tst_label = [tstdata['labelList'][vin] for vin in tstvin]

trdata = loader.svm_feature_loader([],trvin,trainDataFolder)['testset']
tr_feature = [trdata['featureList'][vin] for vin in trvin]
tr_label = [trdata['labelList'][vin] for vin in trvin]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(tr_feature,tr_label)
predict = knn.predict(tst_featrure)
diff = map(abs,predict-tst_label)

accuracy = 1-sum(diff)/float(len(diff))
print accuracy
