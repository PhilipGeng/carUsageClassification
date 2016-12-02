import loader
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
from sklearn.lda import LDA as LinearDiscriminantAnalysis

dataFolder = "data_all"
modelFolder = "5foldall"
labels = np.asarray(loader.label_loader(dataFolder).keys())
kf=KFold(n=len(labels),n_folds=5,shuffle=True)
pca_score=0.0
nopca_score=0.0
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

    pca = LinearDiscriminantAnalysis(n_components=7)
    svm_tr_set_feature_pca = pca.fit_transform(svm_tr_set_feature,svm_tr_set_label)

    #transform test set feature
    svm_tst_set_feature_pca = pca.transform(svm_tst_set_feature)

    #Normalization
    feat_mean = np.mean(svm_tr_set_feature_pca, axis=0)
    feat_std = np.std(svm_tr_set_feature_pca, axis=0)
    svm_tr_set_feature_pca -= feat_mean
    svm_tr_set_feature_pca /= feat_std
    #normalize test data
    svm_tst_set_feature_pca-= feat_mean
    svm_tst_set_feature_pca /= feat_std

    #Training SVM Model
    model = SVC(probability=True)
    model.fit(svm_tr_set_feature_pca, svm_tr_set_label)

    #Measuring training and test accuracy
    tr_accuracy = np.mean(np.argmax(model.predict_proba(svm_tr_set_feature_pca), axis=1) == svm_tr_set_label)
    tst_predicted = model.predict_proba(svm_tst_set_feature_pca)
    tst_accuracy = np.mean(np.argmax(tst_predicted,axis=1)== svm_tst_set_label)
    pca_score += tst_accuracy
    print "svm Train Accuracy:%f, validation Accuracy:%f" % (tr_accuracy, tst_accuracy)
    #no pca
    model = SVC(probability=True)
    model.fit(svm_tr_set_feature, svm_tr_set_label)

    #Measuring training and test accuracy
    tr_accuracy = np.mean(np.argmax(model.predict_proba(svm_tr_set_feature), axis=1) == svm_tr_set_label)
    tst_predicted = model.predict_proba(svm_tst_set_feature)
    tst_accuracy = np.mean(np.argmax(tst_predicted,axis=1)== svm_tst_set_label)
    nopca_score += tst_accuracy
    print "svm Train Accuracy:%f, validation Accuracy:%f" % (tr_accuracy, tst_accuracy)

print pca_score/5
print nopca_score/5