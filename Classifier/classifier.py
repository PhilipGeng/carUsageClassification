import CNN_V1 as cnn
import svm as svm
import loader
from sklearn.cross_validation import KFold
import numpy as np
from sklearn.metrics import auc,roc_curve
import matplotlib.pyplot as plt
from scipy import interp

dataFolder = "data_all"
modelFolder = "5foldall_ROC"
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
labels = np.asarray(loader.label_loader(dataFolder).keys())
kf=KFold(n=len(labels),n_folds=5,shuffle=True)
tprs = []
base_fpr = np.linspace(0, 1, 101)
plt.figure()
svm_th = 0
cnn_th = 0

''''''
for tr,tst in kf:
    trset = labels[tr] #vin
    tstset = labels[tst]#vin
    #process svm data
    svmdata = loader.svm_feature_loader(trset,tstset,dataFolder)
    svm_trainset = svmdata['trainingset']
    svm_testset = svmdata['testset']
    svm_tr_set_feature = svm_trainset['featureList']
    svm_tr_set_label = svm_trainset['labelList']
    svm_tst_set_feature = [svm_testset['featureList'][vin] for vin in tstset]
    svm_tst_set_label = [svm_testset['labelList'][vin] for vin in tstset]

    #process cnn_data
    cnndata = loader.CNN_feature_loader(trset,tstset,dataFolder)
    cnn_train = cnndata['training']
    cnn_test = cnndata['vin_testing']
    cnn_test_extracted = [cnn_test[vin] for vin in tstset]

    cnn_recordTest = cnndata['record_testing']
    cnn_rTdata = np.asarray(map(lambda x:x['data'],cnn_recordTest))
    cnn_rt_length = len(cnn_rTdata)
    cnn_rT_data = cnn_rTdata.reshape(cnn_rt_length,576)
    cnn_rT_label = np.asarray(map(lambda x:x['label'],cnn_recordTest)).reshape(cnn_rt_length,2)
    #print "training set"
    #print trset
    #print "testing set"
    #print tstset

    svm_tst = svm.train(svm_tr_set_feature,svm_tr_set_label,svm_tst_set_feature,svm_tst_set_label,modelFolder,svm_th)
    cnn_tst = cnn.train(cnn_train,cnn_test,cnn_rT_data,cnn_rT_label,modelFolder,cnn_th)
    svm_th = max(svm_th,svm_tst)
    cnn_th = max(cnn_th,cnn_tst)
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

    result_probability = map(lambda x:1-(x[0][0]+x[1][0])/2,compound)
    label_roc = map(lambda x:int(x),svm_tst_set_label)
    fpr, tpr, thresholds = roc_curve(label_roc, result_probability)
    a = auc(fpr,tpr)
    print "AUC: %f"%(a)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % a,alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

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
print "svm accuracy: %f" %((TP_SVM+TN_SVM)/float(len(tstset)*5))
print "CNN: TP: %i, FP: %i, TN: %i, FN: %i"%(TP_CNN,FP_CNN,TN_CNN,FN_CNN)
print "CNN accuracy: %f" %((TP_CNN+TN_CNN)/float(len(tstset)*5))
print "ensembled: TP: %i, FP: %i, TN: %i, FN: %i"%(TP,FP,TN,FN)
print "ensembled accuracy: %f" %((TP+TN)/float(len(tstset)*5))
buildonall = False
if(buildonall):
    trset = labels
    tstset = []
    svmdata = loader.svm_feature_loader(trset,tstset,dataFolder)
    svm_trainset = svmdata['trainingset']
    svm_testset = svmdata['testset']
    svm_tr_set_feature = svm_trainset['featureList']
    svm_tr_set_label = svm_trainset['labelList']
    
    #process cnn_data
    cnndata = loader.CNN_feature_loader(trset,tstset,dataFolder)
    cnn_train = cnndata['training']

    svm_tst = svm.train(svm_tr_set_feature,svm_tr_set_label,svm_tst_set_feature,svm_tst_set_label,modelFolder,svm_th-0.01)
    cnn_tst = cnn.train(cnn_train,cnn_test,cnn_rT_data,cnn_rT_label,modelFolder,cnn_th-0.01)


tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std


plt.plot(base_fpr, mean_tprs, 'b')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
