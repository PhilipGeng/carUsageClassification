import CNN_V1 as cnn
import svm as svm
import loader as loader
from sklearn.cross_validation import KFold
import numpy as np
from sklearn.metrics import auc,roc_curve
import matplotlib.pyplot as plt

modelFolder = "5foldtrain"
dataFolder = "data_test"
tstset = np.asarray(loader.label_loader(dataFolder).keys())
#process svm data
svmdata = loader.svm_feature_loader([], tstset, dataFolder)
svm_testset = svmdata['testset']
svm_tst_set_feature = [svm_testset['featureList'][vin] for vin in tstset]
svm_tst_set_label = [svm_testset['labelList'][vin] for vin in tstset]
numofone =  np.sum(svm_tst_set_label)

#process cnn_data
cnndata = loader.CNN_feature_loader([], tstset, dataFolder)
cnn_test = cnndata['vin_testing']
cnn_test_extracted = [cnn_test[vin] for vin in tstset]

cnn_recordTest = cnndata['record_testing']
cnn_rTdata = np.asarray(map(lambda x:x['data'],cnn_recordTest))
cnn_rt_length = len(cnn_rTdata)
cnn_rT_data = cnn_rTdata.reshape(cnn_rt_length,576)
cnn_rT_label = np.asarray(map(lambda x:x['label'],cnn_recordTest)).reshape(cnn_rt_length,2)
print "testing set length %i" %(len(tstset))
print "label ratio: %i : %i" %(len(tstset)-numofone,numofone)


print "=========testing phase========="
s = svm.classify(svm_tst_set_feature,modelFolder)
c = cnn.classify("trained/"+modelFolder+"/cnnmodel.ckpt",cnn_test_extracted)
#print "svm prediction: "
#print s
#print "cnn prediction"
#print c

compound = zip(s,c)
result = map(lambda x:1 if(x[0][0]+x[1][0]<1)else 0,compound)
s_res = map(lambda x:1 if(x[0]<0.5)else 0,s)
c_res = map(lambda x:1 if(x[0]<0.5)else 0,c)

TP=0
FP=0
FN=0
TN=0
TP_SVM=0
FP_SVM=0
FN_SVM=0
TN_SVM=0
TP_CNN=0
FP_CNN=0
FN_CNN=0
TN_CNN=0
for i in range(len(result)):
    if(result[i]==svm_tst_set_label[i] and result[i]==0):
        TN+=1
    if(result[i]==svm_tst_set_label[i] and result[i]==1):
        TP+=1
    if(result[i]!=svm_tst_set_label[i] and result[i]==0):
        FN+=1
    if(result[i]!=svm_tst_set_label[i] and result[i]==1):
        FP+=1
    if(s_res[i]==svm_tst_set_label[i] and s_res[i]==0):
        TN_SVM+=1
    if(s_res[i]==svm_tst_set_label[i] and s_res[i]==1):
        TP_SVM+=1
    if(s_res[i]!=svm_tst_set_label[i] and s_res[i]==0):
        FN_SVM+=1
    if(s_res[i]!=svm_tst_set_label[i] and s_res[i]==1):
        FP_SVM+=1
    if(c_res[i]==svm_tst_set_label[i] and c_res[i]==0):
        TN_CNN+=1
    if(c_res[i]==svm_tst_set_label[i] and c_res[i]==1):
        TP_CNN+=1
    if(c_res[i]!=svm_tst_set_label[i] and c_res[i]==0):
        FN_CNN+=1
    if(c_res[i]!=svm_tst_set_label[i] and c_res[i]==1):
        FP_CNN+=1
print "svm: TP: %i, FP: %i, TN: %i, FN: %i"%(TP_SVM,FP_SVM,TN_SVM,FN_SVM)
print "svm accuracy: %f" %((TP_SVM+TN_SVM)/float(len(tstset)))
print "CNN: TP: %i, FP: %i, TN: %i, FN: %i"%(TP_CNN,FP_CNN,TN_CNN,FN_CNN)
print "CNN accuracy: %f" %((TP_CNN+TN_CNN)/float(len(tstset)))
print "ensembled: TP: %i, FP: %i, TN: %i, FN: %i"%(TP,FP,TN,FN)
print "ensembled accuracy: %f" %((TP+TN)/float(len(tstset)))


result_probability = map(lambda x:1-(x[0][0]+x[1][0])/2,compound)
label_roc = map(lambda x:int(x),svm_tst_set_label)
fpr, tpr, thresholds = roc_curve(label_roc, result_probability)
a = auc(fpr,tpr)
print "AUC: %f"%(a)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % a)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
