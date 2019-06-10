import os
import numpy as np
import tensorflow as tf
import input_data
import model
import time

from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(train):
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([300, 300])
    image = np.array(image)
    return image


def evaluate_one_image():
    train_dir = "/home/sxy/PycharmProjects/defect2/data/test_less"
    train, train_label = input_data.get_files(train_dir)
    image_array = get_one_image(train)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(image_array, tf.float32)
        image = tf.reshape(image, [1, 300, 300, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[300, 300, 3])

        logs_train_dir = "/home/sxy/PycharmProjects/defect2/logs/train-1"
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Reading checkpoints...")

            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loading success, global_step is %s" % global_step)
            else:
                print("No checkpoint file found")
            start = time.clock()
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)

            end=time.clock()

            print(end-start)
            if max_index == 0:
                print("This is a perfect with possibility %.6f" % prediction[:, 0])
            else:
                print("This is a defect with possibility %.6f" % prediction[:, 1])



evaluate_one_image()