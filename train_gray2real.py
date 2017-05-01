from model import *
LR = 0.0002
def train():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_dir = CURRENT_DIR + '/logs/'
    address = input("Enter the training data location: ")
    name = input("Enter the name of save files: ")
    images_r = inputs_edge2real(SIZE,address)
    images = tf.placeholder(tf.float32, [SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 3])
    images_g = tf.placeholder(tf.float32, [SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 1])
    images_color = tf.placeholder(tf.float32, [SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 3])
    z = tf.placeholder(tf.float32, [SIZE, Z_DIM], name='z')
    Pre_z, e1, e2, e3, e4 = generator_encoder(images_g, images_color, SIZE)
    G_e = generator_decoder(Pre_z, e1, e2, e3, e4, SIZE)
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(G_e, images))
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]
    saver = tf.train.Saver()
    g_d_optim = tf.train.AdamOptimizer(LR, beta1 = 0.5) \
        .minimize(g_loss, var_list=g_vars, global_step=global_step)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    init = tf.initialize_all_variables()
    sess.run(init)
    #saver.restore(sess, "./logs/shoe_gray2bags.ckpt")
    start = 0
    for epoch in range(5):
        batch_idxs = 1001
        if epoch:
            start = 0
        for idx in range(start, batch_idxs):
            sample_base = sess.run(images_r)
            sample_ga, sample_color = build_base_gray2real(sample_base)
            sample_base = sample_base[:,:,128:256,:]
            sample_base = sample_base/255.0
            sess.run([g_d_optim], feed_dict={images: sample_base,images_g:sample_ga, images_color: sample_color})
            if idx % 20 == 0:
                errG = g_loss.eval(
                    {images: sample_base, images_g: sample_ga, images_color: sample_color})
                print("[%4d/%4d] g_loss: %.8f" \
                      % (idx, batch_idxs, errG))

            if idx % 500 == 0:
                sample = sess.run(tf.nn.sigmoid(G_e), feed_dict={images:sample_base,images_g:sample_ga, images_color: sample_color})
                samples_path = CURRENT_DIR + '/samples/'
                save_images(sample_base, [8, 8],
                            samples_path + \
                            'sample_%d_epoch_%d_o.png' % (epoch, idx))
                save_images(sample, [8, 8],
                            samples_path + \
                            'sample_%d_epoch_%d.png' % (epoch, idx))

                print
                '\n' * 2
                print('===========    %d_epoch_%d.png save down    ==========='
                      % (epoch, idx))
                print
                '\n' * 2

            if (idx % 100 == 0) or (idx + 1 == batch_idxs):
                checkpoint_path = os.path.join(train_dir,
                                               name+'_gray2real.ckpt')
                saver.save(sess, checkpoint_path)
                print
                '*********    model saved    *********'

        print
        '******* start with %d *******' % start

    coord.request_stop()
    coord.join(threads)
    sess.close()
if __name__ == '__main__':
    train()