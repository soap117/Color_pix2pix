from model import *
LR = 0.0002
def train():
    SIZE = 1
    name = input("Enter the name of save files: ")
    p1 = input("Enter the name of line picture: ")
    p2 = input("Enter the name of color picture: ")
    train_dir = CURRENT_DIR + '/logs/'
    ke = tf.placeholder(tf.float32)
    images_line = tf.placeholder(tf.float32, [SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 1])
    images_color = tf.placeholder(tf.float32, [SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 3])
    Pre_z, e1, e2, e3, e4 = s_generator_encoder(images_line, images_color, SIZE)
    G = s_generator_decoder(Pre_z, e1, e2, e3, e4, SIZE, keep_prob=ke)
    Pre_z, e1, e2, e3, e4 = generator_encoder(tf.nn.sigmoid(G), images_color, SIZE)
    G_e = generator_decoder(Pre_z, e1, e2, e3, e4, SIZE, keep_prob=ke)
    t_vars = tf.trainable_variables()
    s_g_vars = [var for var in t_vars if 's_g_' in var.name]
    g_vars = [var for var in t_vars if 's_' not in var.name]
    saver1 = tf.train.Saver(g_vars)
    saver2 = tf.train.Saver(s_g_vars)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    init = tf.initialize_all_variables()
    sess.run(init)
    saver1.restore(sess, train_dir+name+'_gray2real.ckpt')
    saver2.restore(sess, train_dir+name+'_edge2gray.ckpt')
    images_t = np.zeros((SIZE, 128, 128, 1))
    samples_color = np.zeros((SIZE, 128, 128, 3))
    t = cv.imread(p1)
    t2 = cv.imread(p2)
    t = cv.cvtColor(t, cv.COLOR_BGR2GRAY)
    t2 = cv.cvtColor(t2, cv.COLOR_RGB2BGR)
    images_t[0, :, :, 0] = cv.resize(t, (128, 128)) / 255.0
    temp = cv.resize(t2, (16, 16), interpolation=cv.INTER_AREA)
    samples_color[0] = cv.resize(temp, (128, 128), interpolation=cv.INTER_AREA) / 255.0
    sample, sample_gray = sess.run([tf.nn.sigmoid(G_e), tf.nn.sigmoid(G)],
                                   feed_dict={images_line: images_t, images_color: samples_color, ke: 1.0})
    sample = sample * 255
    sample_gray = sample_gray * 255
    img = Image.fromarray(np.uint8(sample[0]))
    img.show()
    img = Image.fromarray(np.uint8(sample_gray[0,:,:,0]))
    img.show()

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    train()