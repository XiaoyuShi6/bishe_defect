import tensorflow as tf
import numpy as np
import os

# 获取文件路径和标签
def get_files(file_dir):
    # file_dir: 文件夹路径
    # return: 乱序后的图片和标签

    perfects = []
    label_perfects = []
    defects = []
    label_defects = []
    # 载入数据路径并写入标签值
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'perfect':
            perfects.append(file_dir +'/'+ file)
            label_perfects.append(0)
        else:
            defects.append(file_dir +'/'+ file)
            label_defects.append(1)
    print("There are %d 张无缺陷图\nThere are %d 张有缺陷图" % (len(perfects), len(defects)))

    # 打乱文件顺序
    image_list = np.hstack((perfects, defects))
    label_list = np.hstack((label_perfects, label_defects))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()     # 转置
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list

# 生成相同大小的批次
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # image, label: 要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch

    # 将python.list类型转换成tf能够识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 生成队列
    input_queue = tf.train.slice_input_producer([image, label])

    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # 统一图片大小
    # 视频方法
    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 我的方法
    image = tf.image.resize_images(image, image_H, image_W, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    # image = tf.image.per_image_standardization(image)   # 标准化数据
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,   # 线程
                                              capacity=capacity)

    # 这行多余？
    # label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch

# TEST
# import matplotlib.pyplot as plt
#
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 208
# IMG_H = 208
#
# train_dir = "/home/sxy/PycharmProjects/defect2/data/train"
# image_list, label_list = get_files(train_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#     i = 0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     try:
#         while not coord.should_stop() and i < 1:
#             img, label = sess.run([image_batch, label_batch])
#
#             for j in np.arange(BATCH_SIZE):
#                 print("label: %d" % label[j])
#                 plt.imshow(img[j, :, :, :])
#                 plt.show()
#             i += 1
#     except tf.errors.OutOfRangeError:
#         print("done!")
#     finally:
#         coord.request_stop()
#     coord.join(threads)