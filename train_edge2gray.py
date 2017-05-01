from model import *
LR = 0.0002
def train():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_dir = CURRENT_DIR + '/logs/'
    address = input("Enter the training data location: ")
    name = input("Enter the name of save files: ")
    images_r = inputs_edge2real(SIZE, address)
    images = tf.placeholder(tf.float32, [SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 1])
    images_g = tf.placeholder(tf.float32, [SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 1])
    z = tf.placeholder(tf.float32, [SIZE, Z_DIM], name='z')
    Pre_z, e1, e2, e3, e4 = s_generator_encoder(images_g, SIZE)
    G_e = s_generator_decoder(Pre_z, e1, e2, e3, e4, SIZE)
    D, D_logits = s_discriminator(images, images_g, SIZE)
    D_, D_logits_ = s_discriminator(tf.nn.sigmoid(G_e), images_g, SIZE, reuse=True)
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(D_logits_, tf.zeros_like(D_)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(D_logits_, tf.ones_like(D_)))
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]
    saver = tf.train.Saver()
    d_optim = tf.train.AdamOptimizer(LR, beta1=0.5) \
        .minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_d_optim = tf.train.AdamOptimizer(LR, beta1=0.5) \
        .minimize(g_loss, var_list=g_vars, global_step=global_step)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    init = tf.initialize_all_variables()
    sess.run(init)
    start = 0
    #saver.restore(sess, "./logs/shoe_edge2gray.ckpt")
    for epoch in range(10):
        batch_idxs = 1001
        if epoch:
            start = 0

        for idx in range(start, batch_idxs):
            sample_base = sess.run(images_r)
            sample_ga, sample_base = build_base_edge2gray(sample_base)
            errD_fake = d_loss_fake.eval(
                {images: sample_base, images_g: sample_ga})
            errD_real = d_loss_real.eval(
                {images: sample_base, images_g: sample_ga})
            if errD_fake+errD_real>1.0:
                sess.run([d_optim], feed_dict={images: sample_base,images_g:sample_ga})
            sess.run([g_d_optim], feed_dict={images:sample_base,images_g:sample_ga})
            if idx % 20 == 0:
                errG = g_loss.eval(
                    {images: sample_base, images_g: sample_ga})
                print("[%4d/%4d] d_loss: %.8f, g_loss: %.8f" \
                      % (idx, batch_idxs, errD_fake + errD_real, errG))

            if idx % 500 == 0:
                sample = sess.run(tf.nn.sigmoid(G_e), feed_dict={images:sample_base,images_g:sample_ga})
                samples_path = CURRENT_DIR + '/samples/'
                save_images(sample_base, [8, 8],
                            samples_path + \
                            'sample_%d_epoch_%d_o.png' % (epoch, idx))
                save_images(sample_ga, [8, 8],
                            samples_path + \
                            'sample_%d_epoch_%d_g.png' % (epoch, idx))
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
                                               name+'_edge2gray.ckpt')
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