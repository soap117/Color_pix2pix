from ops import *
import os
import cv2 as cv
GF = 64
DF = 64
import PIL.Image as Image
import scipy.misc
OUTPUT_SIZE = 128
CURRENT_DIR = os.getcwd()
Z_DIM = 256
SIZE = 64
def generator_decoder(z,e1,e2,e3,e4,size,is_train=True, name='generator_decoder', reuse=False, keep_prob = 0.5):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    s2, s4, s8, s16 = \
        int(OUTPUT_SIZE / 2), int(OUTPUT_SIZE / 4), int(OUTPUT_SIZE / 8),int(OUTPUT_SIZE / 16)
    h1 = deconv2d(z, [size, s16, s16, GF * 8], name='g_d_deconv2d0')
    h1 = tf.concat(3, [h1, e4])
    h1 = relu(batch_norm_layer(h1, name='g_d_bn1', is_train=is_train))
    h1 = tf.nn.dropout(h1, keep_prob)
    h2 = deconv2d(h1, [size, s8, s8, GF * 4], name='g_d_deconv2d1')
    h2 = tf.concat(3, [h2, e3])
    h2 = relu(batch_norm_layer(h2, name='g_d_bn2', is_train=is_train))
    h2 = tf.nn.dropout(h2, keep_prob)
    h3 = deconv2d(h2, [size, s4, s4, GF * 2], name='g_d_deconv2d2')
    h3 = tf.concat(3, [h3, e2])
    h3 = relu(batch_norm_layer(h3, name='g_d_bn3', is_train=is_train))
    h4 = deconv2d(h3, [size, s2, s2, GF * 1], name='g_d_deconv2d3')
    h4 = tf.concat(3, [h4, e1])
    h4 = relu(batch_norm_layer(h4, name='g_d_bn4', is_train=is_train))
    h5 = deconv2d(h4, [size, OUTPUT_SIZE, OUTPUT_SIZE, 3],
                  name='g_d_deconv2d4')
    return h5
def s_generator_decoder(z,e1,e2,e3,e4,size,is_train=True, name='generator_decoder', reuse=False, keep_prob = 0.5):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    s2, s4, s8, s16 = \
        int(OUTPUT_SIZE / 2), int(OUTPUT_SIZE / 4), int(OUTPUT_SIZE / 8),int(OUTPUT_SIZE / 16)
    h1 = deconv2d(z, [size, s16, s16, GF * 8], name='s_g_d_deconv2d0')
    h1 = relu(batch_norm_layer(h1, name='s_g_d_bn1', is_train=is_train))
    h1 = tf.nn.dropout(h1,keep_prob)
    h2 = deconv2d(h1, [size, s8, s8, GF * 4], name='s_g_d_deconv2d1')
    h2 = tf.concat(3, [h2, e3])
    h2 = relu(batch_norm_layer(h2, name='s_g_d_bn2', is_train=is_train))
    h2 = tf.nn.dropout(h2, keep_prob)
    h3 = deconv2d(h2, [size, s4, s4, GF * 2], name='s_g_d_deconv2d2')
    h3 = tf.concat(3, [h3, e2])
    h3 = relu(batch_norm_layer(h3, name='s_g_d_bn3', is_train=is_train))
    h4 = deconv2d(h3, [size, s2, s2, GF * 1], name='s_g_d_deconv2d3')
    h4 = tf.concat(3, [h4, e1])
    h4 = relu(batch_norm_layer(h4, name='s_g_d_bn4', is_train=is_train))
    h5 = deconv2d(h4, [size, OUTPUT_SIZE, OUTPUT_SIZE, 1],
                  name='s_g_d_deconv2d4')
    return h5

def generator_encoder(x, color, size,is_train=True, reuse=False,name='generator_encoder'):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    x = tf.concat(3, [x, color])
    h5_o = conv2d(x, GF,name='g_e_conv2d5')
    h5 = relu(batch_norm_layer(h5_o, name='g_e_bn5', is_train=is_train))
    h4_o = conv2d(h5, GF * 2, name='g_e_conv2d4')
    h4 = relu(batch_norm_layer(h4_o, name='g_e_bn4', is_train=is_train))
    h3_o = conv2d(h4, GF * 4, name='g_e_conv2d3')
    h3 = relu(batch_norm_layer(h3_o, name='g_e_bn3', is_train=is_train))
    h2_o = conv2d(h3, GF * 8, name='g_e_conv2d2')
    h2 = relu(batch_norm_layer(h2_o, name='g_e_bn2', is_train=is_train))
    h1_o = conv2d(h2, GF * 8, name='g_e_conv2d1')
    z = relu(batch_norm_layer(h1_o, name='g_e_bn1', is_train=is_train))
    return z, h5_o, h4_o, h3_o, h2_o

def s_generator_encoder(x, size,is_train=True, reuse=False,name='generator_encoder'):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    h5_o = conv2d(x, GF,name='s_g_e_conv2d5')
    h5 = relu(batch_norm_layer(h5_o, name='s_g_e_bn5', is_train=is_train))
    h4_o = conv2d(h5, GF * 2, name='s_g_e_conv2d4')
    h4 = relu(batch_norm_layer(h4_o, name='s_g_e_bn4', is_train=is_train))
    h3_o = conv2d(h4, GF * 4, name='s_g_e_conv2d3')
    h3 = relu(batch_norm_layer(h3_o, name='s_g_e_bn3', is_train=is_train))
    h2_o = conv2d(h3, GF * 8, name='s_g_e_conv2d2')
    h2 = relu(batch_norm_layer(h2_o, name='s_g_e_bn2', is_train=is_train))
    h1_o = conv2d(h2, GF * 8, name='s_g_e_conv2d1')
    z = relu(batch_norm_layer(h1_o, name='s_g_e_bn1', is_train=is_train))
    return z, h5_o, h4_o, h3_o, h2_o

def discriminator(x_d, color, size,reuse=False, name='discriminator'):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    x2 = tf.concat(3, [x_d, color])
    h0 = lrelu(conv2d(x2, DF, name='d_h0_conv'), name='d_h0_lrelu')
    h1 = lrelu(batch_norm_layer(conv2d(h0, DF * 2, name='d_h1_conv'),
                          name='d_h1_bn'), name='d_h1_lrelu')
    h2 = lrelu(batch_norm_layer(conv2d(h1, DF * 4, name='d_h2_conv'),
                          name='d_h2_bn'), name='d_h2_lrelu')
    h3 = lrelu(batch_norm_layer(conv2d(h2, DF * 8, name='d_h3_conv'),
                          name='d_h3_bn'), name='d_h3_lrelu')
    h4 = fully_connected(tf.reshape(h3, [size, -1]), 1, 'd_h4_fc')

    return tf.nn.sigmoid(h4), h4

def s_discriminator(x_d, x_b, size,reuse=False, name='discriminator'):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    x2 = tf.concat(3, [x_d, x_b])
    h0 = lrelu(conv2d(x2, DF, name='d_h0_conv'), name='d_h0_lrelu')
    h1 = lrelu(batch_norm_layer(conv2d(h0, DF * 2, name='d_h1_conv'),
                          name='d_h1_bn'), name='d_h1_lrelu')
    h2 = lrelu(batch_norm_layer(conv2d(h1, DF * 4, name='d_h2_conv'),
                          name='d_h2_bn'), name='d_h2_lrelu')
    h3 = lrelu(batch_norm_layer(conv2d(h2, DF * 8, name='d_h3_conv'),
                          name='d_h3_bn'), name='d_h3_lrelu')
    h4 = fully_connected(tf.reshape(h3, [size, -1]), 1, 'd_h4_fc')

    return tf.nn.sigmoid(h4), h4

def read_and_decode(filename_queue):
    """
    read and decode tfrecords
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'image_raw': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['image_raw'], tf.uint8)

    image = tf.reshape(image, [OUTPUT_SIZE, 2*OUTPUT_SIZE, 3])
    image = tf.cast(image, tf.float32)
    image = image

    return image

def inputs(data_dir, batch_size, name='input'):
    """
    Reads input data num_epochs times.
    """
    with tf.name_scope(name):
        filenames = [
            os.path.join(data_dir, '%d.tfrecords' % ii) for ii in range(12)]
        filename_queue = tf.train.string_input_producer(filenames)

        image = read_and_decode(filename_queue)

        images = tf.train.shuffle_batch([image], batch_size=batch_size,capacity=2000,
                                                min_after_dequeue=1000)
        return images

def inputs_edge2real(batch_size, name='input', data_dir = CURRENT_DIR):
    """
    Reads input data num_epochs times.
    """
    with tf.name_scope(name):
        filenames = [
            os.path.join(data_dir, '%d.tfrecords' % ii) for ii in range(10)]
        filename_queue = tf.train.string_input_producer(filenames)

        image = read_and_decode_edge2shoes(filename_queue)

        images = tf.train.shuffle_batch([image], batch_size=batch_size,capacity=2000,
                                                min_after_dequeue=1000)
        return images

def save_images(images, size, path):
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image
    return scipy.misc.imsave(path, merge_img)

def build_base_gray2real(sample_base, SIZE=64):
    sample_ga = np.zeros((SIZE, OUTPUT_SIZE, OUTPUT_SIZE,1))
    sample_color = np.zeros((SIZE, 6, 6, 3))
    sample_color_2= np.zeros((SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 3))
    for i in range(SIZE):
        sample_ga[i,:,:,0] = cv.cvtColor(sample_base[i,:,128:256,:], cv.COLOR_BGR2GRAY)
        t = sample_base[i,:,128:256,:]
        sample_color[i] = cv.resize(t, (6, 6), interpolation=cv.INTER_AREA)
        sample_color_2[i] = cv.resize(sample_color[i], (128, 128), interpolation=cv.INTER_AREA)
    return sample_ga/255.0, sample_color_2/255.0

def build_base_edge2gray(sample_base, SIZE=64):
    sample_ga = np.zeros((SIZE, OUTPUT_SIZE, OUTPUT_SIZE,1))
    sample_color_2= np.zeros((SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 1))
    for i in range(SIZE):
        sample_ga[i,:,:,0] = cv.cvtColor(sample_base[i,:,0:128,:], cv.COLOR_BGR2GRAY)
        sample_color_2[i,:,:,0] = cv.cvtColor(sample_base[i,:,128:256,:], cv.COLOR_BGR2GRAY)
    return sample_ga/255.0, sample_color_2/255.0