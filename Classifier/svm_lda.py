from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
import numpy as np
from sklearn.externals import joblib

def train(tr_set_feature,tr_set_label,tst_set_feature,tst_set_label,modelPath,saveThreshold):
    lda_model_path = "trained/"+modelPath+"/lda.pkl"
    normalization_path="trained/"+modelPath+"/norm.pkl"
    svm_model_path = "trained/"+modelPath+"/svm.pkl"

    #build lda use tr set feature

    lda = LDA(solver="svd",n_components=3)
    lda.fit(tr_set_feature, tr_set_label)

    print lda.explained_variance_ratio_
    tr_set_feature = lda.transform(tr_set_feature)
    #save lda
    joblib.dump(lda, lda_model_path)

    #transform test set feature
    tst_set_feature = lda.transform(tst_set_feature)
    print tst_set_feature
    #build pca use tr set feature
    #pca = PCA(n_components=7)
    #tr_set_feature = pca.fit_transform(tr_set_feature)
    #print "pca explains variance ratio: "
    #print sum(pca.explained_variance_ratio_)
    #print pca.explained_variance_ratio_

    #save pca
    #joblib.dump(pca, pca_model_path)

    #transform test set feature
    #tst_set_feature = pca.transform(tst_set_feature)

    #Normalization
    feat_mean = np.mean(tr_set_feature,axis=0)
    feat_std = np.std(tr_set_feature,axis=0)
    tr_set_feature -= feat_mean
    tr_set_feature /= feat_std
    #save normalization
    content = {"mean":feat_mean,"std":feat_std}
    joblib.dump(content,normalization_path)
    #normalize test data
    tst_set_feature -= feat_mean
    tst_set_feature /= feat_std

    #Training SVM Model
    model = SVC(probability=True)
    model.fit(tr_set_feature, tr_set_label)

    #Measuring training and test accuracy
    tr_accuracy = np.mean(np.argmax(model.predict_proba(tr_set_feature),axis=1) == tr_set_label)
    tst_predicted = model.predict_proba(tst_set_feature)
    tst_accuracy = np.mean(np.argmax(tst_predicted,axis=1)== tst_set_label)

    print "svm Train Accuracy:%f, validation Accuracy:%f" % (tr_accuracy, tst_accuracy)
    if(tst_accuracy>saveThreshold):
        joblib.dump(model, svm_model_path)
    return tst_accuracy

def classify(data,modelPath):
    lda_model_path = "trained/"+modelPath+"/lda.pkl"
    #pca_model_path = "trained/pca.pkl"
    normalization_path="trained/"+modelPath+"/norm.pkl"
    svm_model_path = "trained/"+modelPath+"/svm.pkl"

    lda = joblib.load(lda_model_path)
    #pca = joblib.load(pca_model_path)

    normalization = joblib.load(normalization_path)
    feat_mean = normalization['mean']
    feat_std = normalization['std']

    svm = joblib.load(svm_model_path)

    features = lda.transform(data)
    #features = pca.transform(data)

    features -= feat_mean
    features /= feat_std
    return svm.predict_proba(features)


