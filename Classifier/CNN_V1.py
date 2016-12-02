
import numpy as np
import tensorflow as tf

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_3x3(x):
    return tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

def train(train,test,rTdata,rTlabel,modelPath,saveThreshold):
#model construction
    num_of_epoch=25
    #data = loader.matrix_loader(trainVin,testVin)
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, 576])
    y_ = tf.placeholder(tf.float32, [None, 2])

    keep_prob = tf.placeholder(tf.float32)
    y_conv = model(x,keep_prob)

    # model training
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    y_predicted = tf.reduce_mean(y_conv,0)

    correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        #training procedure
        for i in range(num_of_epoch):
            trset = train
            #batch size = 1
            for each in trset:
                train_data = each['data']
                label_data = each['label']
                train_step.run(feed_dict = {x: train_data, y_: label_data, keep_prob: 0.5})
            # test of each vin
            if i%10==9:
                print "training epoch"+str(i+1)
                correct = 0
                for t in test.keys():
                    onetestdata = test[t]
                    l = len(onetestdata['data'])
                    testData = np.asarray(onetestdata['data']).reshape(l,576)
                    testLabel = np.asarray(onetestdata['label']).reshape(1,2)
                    predicted = y_predicted.eval(feed_dict = {x:testData, keep_prob: 1.0})
                    if(np.argmax(predicted)==np.argmax(testLabel)):
                        correct+=1
                vin_acc = float(correct)/float(len(test.keys()))
                print "cnn vin based accuracy:"+str(vin_acc)
                # test of each record
                print("cnn records based accuracy %g"%(accuracy.eval(feed_dict={x: rTdata, y_: rTlabel, keep_prob: 1.0})))
            # save model
        if(vin_acc>saveThreshold):
            save_path = saver.save(sess, "trained/"+modelPath+"/cnnmodel.ckpt")
            print "cnn: Model saved in file: ", save_path
    return vin_acc

def model(x,keep_prob):
    W_conv1 = weight_varible([5, 5, 1, 5])
    b_conv1 = bias_variable([5])

    # conv layer-1
    x_image = tf.reshape(x, [-1, 24, 24, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = avg_pool_2x2(h_conv1)

    # conv layer-2
    W_conv2 = weight_varible([5, 5, 5, 15])
    b_conv2 = bias_variable([15])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = avg_pool_3x3(h_conv2)

    # full connection
    W_fc1 = weight_varible([60, 30])
    b_fc1 = bias_variable([30])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 60])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # output layer:
    W_fc2 = weight_varible([30, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv

def classify(modelpath,test):
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, 576])

    keep_prob = tf.placeholder(tf.float32)
    y_conv = model(x,keep_prob)

    y_predicted = tf.reduce_mean(y_conv,0)

    with tf.Session() as sess:
    # Restore variables from disk.
        saver = tf.train.Saver()
        saver.restore(sess, modelpath)
        # test of each vin
        predictedLabels = []
        for onetestdata in test:
            l = len(onetestdata['data'])
            testData = np.asarray(onetestdata['data']).reshape(l,576)
            predicted = y_predicted.eval(feed_dict = {x:testData, keep_prob: 1.0})
            predictedLabels.append(predicted)
    return predictedLabels