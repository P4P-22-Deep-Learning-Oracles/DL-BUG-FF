# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:17:55 2021

@author: Michael

Buggy code sample 3 as on excel
"""
with tf.Session() as sess:
model_saver = tf.train.import_meta_graph(model_save_folder + '/my-model.meta')
model_saver.restore(sess, model_save_folder + '/my-model')
x = tf.placeholder('float')
output = tf.get_collection("output")[0] #output will be the tensor for model's last layer
print("Model restored.")
print('Initialized')
#print(sess.run(tf.get_default_graph().get_tensor_by_name('w_conv1:0')))

#collect list of preprocessed data on submission set
inputData = []
with open('stage1_sample_submission.csv') as f:
    reader = csv.reader(f)
    num = 0

    for row in reader:
        if num > 0:
            patient = row[0]
            #print(patient)
            inputData.append(process_data(patient, img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT))
        num += 1

#prediction!
prediction = sess.run(output, feed_dict={x: inputData})
print(prediction)