import os
import numpy as np
import tensorflow as tf
import input_data
import model

N_CLASSES = 2
IMG_H = 300
IMG_W = 300
BATCH_SIZE = 20
CAPACITY = 2000
MAX_STEP = 1000
learning_rate = 0.0001


def run_training():
    train_dir = "/home/sxy/PycharmProjects/defect2/data/train"
    logs_train_dir = "/home/sxy/PycharmProjects/defect2/logs/train-4"
    tf.reset_default_graph()
    train, train_label = input_data.get_files(train_dir)
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.trainning(train_loss, learning_rate)
    train_acc = model.evaluation(train_logits, train_label_batch)
    #测试准确率
    # test_dir="/home/sxy/PycharmProjects/defect2/data/test"
    # test,test_label=input_data.get_files(test_dir)
    # test_batch,test_label_batch=input_data.get_batch(test,
    #                                                       test_label,
    #                                                       IMG_W,
    #                                                       IMG_H,
    #                                                       BATCH_SIZE,
    #                                                       CAPACITY)
    # test_logits = model.inference(test_batch, BATCH_SIZE, N_CLASSES)
    # train_loss = model.losses(test_logits, test_label_batch)
    # train_op = model.trainning(train_loss, learning_rate)
    # train_acc = model.evaluation(test_logits, test_label_batch)

    summary_op = tf.merge_all_summaries()
    sess = tf.Session()
    train_writer = tf.train.SummaryWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

            if step % 50 == 0:
                print("Step %d, train loss = %.2f, train accuracy = %.2f%%" % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print("Done training -- epoch limit reached.")
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

# 评估模型
# from PIL import Image
# import matplotlib.pyplot as plt
#
#
# def get_one_image(train):
#     n = len(train)
#     ind = np.random.randint(0, n)
#     img_dir = train[ind]
#
#     image = Image.open(img_dir)
#     plt.imshow(image)
#     plt.show()
#     image = image.resize([208, 208])
#     image = np.array(image)
#     return image
#
#
# def evaluate_one_image():
#     train_dir = "/home/sxy/PycharmProjects/defect2/data/train"
#     train, train_label = input_data.get_files(train_dir)
#     image_array = get_one_image(train)
#
#     with tf.Graph().as_default():
#         BATCH_SIZE = 1
#         N_CLASSES = 2
#
#         image = tf.cast(image_array, tf.float32)
#         image = tf.reshape(image, [1, 208, 208, 3])
#         logit = model.inference(image, BATCH_SIZE, N_CLASSES)
#         logit = tf.nn.softmax(logit)
#
#         x = tf.placeholder(tf.float32, shape=[208, 208, 3])
#
#         logs_train_dir = "/home/sxy/PycharmProjects/defect2/logs/train"
#         saver = tf.train.Saver()
#
#         with tf.Session() as sess:
#             print("Reading checkpoints...")
#             ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#             if ckpt and ckpt.model_checkpoint_path:
#                 global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
#                 saver.restore(sess, ckpt.model_checkpoint_path)
#                 print("Loading success, global_step is %s" % global_step)
#             else:
#                 print("No checkpoint file found")
#
#             prediction = sess.run(logit, feed_dict={x: image_array})
#             max_index = np.argmax(prediction)
#             if max_index == 0:
#                 print("This is a cat with possibility %.6f" % prediction[:, 0])
#             else:
#                 print("This is a dog with possibility %.6f" % prediction[:, 1])


run_training()
#evaluate_one_image()

