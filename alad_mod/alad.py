import sys
import time
import logging

import tensorflow as tf
import sklearn
import numpy as np

from core.skeleton import *

logger = logging.getLogger("ALAD")


class ALAD(AbstractAnomalyDetector):
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess

        # Parameters
        starting_lr = config.learning_rate
        batch_size = config.batch_size
        latent_dim = config.latent_dim
        ema_decay = config.ema_decay

        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Placeholders
        x_pl = tf.placeholder(tf.float32, shape=[None, config.input_dim],
                              name="input_x")
        z_pl = tf.placeholder(tf.float32, shape=[None, latent_dim],
                              name="input_z")
        is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
        learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

        # models
        gen = config.decoder
        enc = config.encoder
        dis_xz = config.discriminator_xz
        dis_xx = config.discriminator_xx
        dis_zz = config.discriminator_zz

        # compile models
        with tf.variable_scope('encoder_model'):
            z_gen = enc(x_pl, is_training=is_training_pl,
                        do_spectral_norm=config.do_spectral_norm)

        with tf.variable_scope('generator_model'):
            x_gen = gen(z_pl, is_training=is_training_pl)
            rec_x = gen(z_gen, is_training=is_training_pl, reuse=True)

        with tf.variable_scope('encoder_model'):
            rec_z = enc(x_gen, is_training=is_training_pl, reuse=True,
                        do_spectral_norm=config.do_spectral_norm)

        with tf.variable_scope('discriminator_model_xz'):
            l_encoder, inter_layer_inp_xz = dis_xz(x_pl, z_gen,
                                                   is_training=is_training_pl,
                                                   do_spectral_norm=config.do_spectral_norm)
            l_generator, inter_layer_rct_xz = dis_xz(x_gen, z_pl,
                                                     is_training=is_training_pl,
                                                     reuse=True,
                                                     do_spectral_norm=config.do_spectral_norm)

        with tf.variable_scope('discriminator_model_xx'):
            x_logit_real, inter_layer_inp_xx = dis_xx(x_pl, x_pl,
                                                      is_training=is_training_pl,
                                                      do_spectral_norm=config.do_spectral_norm)
            x_logit_fake, inter_layer_rct_xx = dis_xx(x_pl, rec_x, is_training=is_training_pl,
                                                      reuse=True, do_spectral_norm=config.do_spectral_norm)

        with tf.variable_scope('discriminator_model_zz'):
            z_logit_real, _ = dis_zz(z_pl, z_pl, is_training=is_training_pl,
                                     do_spectral_norm=config.do_spectral_norm)
            z_logit_fake, _ = dis_zz(z_pl, rec_z, is_training=is_training_pl,
                                     reuse=True, do_spectral_norm=config.do_spectral_norm)

        with tf.name_scope('loss_functions'):
            # discriminator xz
            loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(l_encoder), logits=l_encoder))
            loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(l_generator), logits=l_generator))
            dis_loss_xz = loss_dis_gen + loss_dis_enc

            # discriminator xx
            x_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_real, labels=tf.ones_like(x_logit_real))
            x_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_fake, labels=tf.zeros_like(x_logit_fake))
            dis_loss_xx = tf.reduce_mean(x_real_dis + x_fake_dis)

            # discriminator zz
            z_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_real, labels=tf.ones_like(z_logit_real))
            z_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_fake, labels=tf.zeros_like(z_logit_fake))
            dis_loss_zz = tf.reduce_mean(z_real_dis + z_fake_dis)

            loss_discriminator = dis_loss_xz + dis_loss_xx + dis_loss_zz if \
                config.allow_zz else dis_loss_xz + dis_loss_xx

            # generator and encoder
            gen_loss_xz = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(l_generator), logits=l_generator))
            enc_loss_xz = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(l_encoder), logits=l_encoder))
            x_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_real, labels=tf.zeros_like(x_logit_real))
            x_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_fake, labels=tf.ones_like(x_logit_fake))
            z_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_real, labels=tf.zeros_like(z_logit_real))
            z_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_fake, labels=tf.ones_like(z_logit_fake))

            cost_x = tf.reduce_mean(x_real_gen + x_fake_gen)
            cost_z = tf.reduce_mean(z_real_gen + z_fake_gen)

            cycle_consistency_loss = cost_x + cost_z if config.allow_zz else cost_x
            loss_generator = gen_loss_xz + cycle_consistency_loss
            loss_encoder = enc_loss_xz + cycle_consistency_loss

        with tf.name_scope('optimizers'):
            # control op dependencies for batch norm and trainable variables
            tvars = tf.trainable_variables()
            dxzvars = [var for var in tvars if 'discriminator_model_xz' in var.name]
            dxxvars = [var for var in tvars if 'discriminator_model_xx' in var.name]
            dzzvars = [var for var in tvars if 'discriminator_model_zz' in var.name]
            gvars = [var for var in tvars if 'generator_model' in var.name]
            evars = [var for var in tvars if 'encoder_model' in var.name]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
            update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
            update_ops_dis_xz = [x for x in update_ops if
                                 ('discriminator_model_xz' in x.name)]
            update_ops_dis_xx = [x for x in update_ops if
                                 ('discriminator_model_xx' in x.name)]
            update_ops_dis_zz = [x for x in update_ops if
                                 ('discriminator_model_zz' in x.name)]

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               beta1=0.5)

            with tf.control_dependencies(update_ops_gen):
                gen_op = optimizer.minimize(loss_generator, var_list=gvars,
                                            global_step=global_step)
            with tf.control_dependencies(update_ops_enc):
                enc_op = optimizer.minimize(loss_encoder, var_list=evars)

            with tf.control_dependencies(update_ops_dis_xz):
                dis_op_xz = optimizer.minimize(dis_loss_xz, var_list=dxzvars)

            with tf.control_dependencies(update_ops_dis_xx):
                dis_op_xx = optimizer.minimize(dis_loss_xx, var_list=dxxvars)

            with tf.control_dependencies(update_ops_dis_zz):
                dis_op_zz = optimizer.minimize(dis_loss_zz, var_list=dzzvars)

            # Exponential Moving Average for inference
            def train_op_with_ema_dependency(vars, op):
                ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay)
                maintain_averages_op = ema.apply(vars)
                with tf.control_dependencies([op]):
                    train_op = tf.group(maintain_averages_op)
                return train_op, ema

            train_gen_op, gen_ema = train_op_with_ema_dependency(gvars, gen_op)
            train_enc_op, enc_ema = train_op_with_ema_dependency(evars, enc_op)
            train_dis_op_xz, xz_ema = train_op_with_ema_dependency(dxzvars,
                                                                   dis_op_xz)
            train_dis_op_xx, xx_ema = train_op_with_ema_dependency(dxxvars,
                                                                   dis_op_xx)
            train_dis_op_zz, zz_ema = train_op_with_ema_dependency(dzzvars,
                                                                   dis_op_zz)

        with tf.variable_scope('encoder_model'):
            z_gen_ema = enc(x_pl, is_training=is_training_pl,
                            getter=get_getter(enc_ema), reuse=True,
                            do_spectral_norm=config.do_spectral_norm)

        with tf.variable_scope('generator_model'):
            rec_x_ema = gen(z_gen_ema, is_training=is_training_pl,
                            getter=get_getter(gen_ema), reuse=True)
            x_gen_ema = gen(z_pl, is_training=is_training_pl,
                            getter=get_getter(gen_ema), reuse=True)

        with tf.variable_scope('discriminator_model_xx'):
            l_encoder_emaxx, inter_layer_inp_emaxx = dis_xx(x_pl, x_pl,
                                                            is_training=is_training_pl,
                                                            getter=get_getter(xx_ema),
                                                            reuse=True,
                                                            do_spectral_norm=config.do_spectral_norm)

            l_generator_emaxx, inter_layer_rct_emaxx = dis_xx(x_pl, rec_x_ema,
                                                              is_training=is_training_pl,
                                                              getter=get_getter(
                                                                  xx_ema),
                                                              reuse=True,
                                                              do_spectral_norm=config.do_spectral_norm)

        with tf.name_scope('Testing'):

            with tf.variable_scope('Scores'):
                score_ch = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(l_generator_emaxx),
                    logits=l_generator_emaxx)
                score_ch = tf.squeeze(score_ch)

                rec = x_pl - rec_x_ema
                rec = tf.contrib.layers.flatten(rec)
                score_l1 = tf.norm(rec, ord=1, axis=1,
                                   keep_dims=False, name='d_loss')
                score_l1 = tf.squeeze(score_l1)

                rec = x_pl - rec_x_ema
                rec = tf.contrib.layers.flatten(rec)
                score_l2 = tf.norm(rec, ord=2, axis=1,
                                   keep_dims=False, name='d_loss')
                score_l2 = tf.squeeze(score_l2)

                inter_layer_inp, inter_layer_rct = inter_layer_inp_emaxx, \
                                                   inter_layer_rct_emaxx
                fm = inter_layer_inp - inter_layer_rct
                fm = tf.contrib.layers.flatten(fm)
                score_fm = tf.norm(fm, ord=config.fm_degree, axis=1,
                                   keep_dims=False, name='d_loss')
                score_fm = tf.squeeze(score_fm)

        if config.enable_sm:

            with tf.name_scope('summary'):
                with tf.name_scope('dis_summary'):
                    tf.summary.scalar('loss_discriminator', loss_discriminator, ['dis'])
                    tf.summary.scalar('loss_dis_encoder', loss_dis_enc, ['dis'])
                    tf.summary.scalar('loss_dis_gen', loss_dis_gen, ['dis'])
                    tf.summary.scalar('loss_dis_xz', dis_loss_xz, ['dis'])
                    tf.summary.scalar('loss_dis_xx', dis_loss_xx, ['dis'])
                    if config.allow_zz:
                        tf.summary.scalar('loss_dis_zz', dis_loss_zz, ['dis'])

                with tf.name_scope('gen_summary'):
                    tf.summary.scalar('loss_generator', loss_generator, ['gen'])
                    tf.summary.scalar('loss_encoder', loss_encoder, ['gen'])
                    tf.summary.scalar('loss_encgen_dxx', cost_x, ['gen'])
                    if config.allow_zz:
                        tf.summary.scalar('loss_encgen_dzz', cost_z, ['gen'])

                with tf.name_scope('img_summary'):
                    heatmap_pl_latent = tf.placeholder(tf.float32,
                                                       shape=(1, 480, 640, 3),
                                                       name="heatmap_pl_latent")
                    sum_op_latent = tf.summary.image('heatmap_latent', heatmap_pl_latent)

                sum_op_dis = tf.summary.merge_all('dis')
                sum_op_gen = tf.summary.merge_all('gen')
                sum_op = tf.summary.merge([sum_op_dis, sum_op_gen])
                sum_op_im = tf.summary.merge_all('image')
                sum_op_valid = tf.summary.merge_all('v')

        self.__dict__.update(locals())

    def compute_fm_scores(self, x):
        feed_dict = {self.x_pl: x,
                     self.z_pl: np.random.normal(size=[x.shape[0], self.config.latent_dim]),
                     self.is_training_pl: False}

        return self.sess.run(self.score_fm, feed_dict=feed_dict)

    def get_anomaly_scores(self, x):
        return self.compute_fm_scores(x)

    def fit(self, x, max_epoch, logdir, evaluator):
        sess = self.sess
        saver = tf.train.Saver(max_to_keep=1000)
        writer = tf.summary.FileWriter(logdir, sess.graph)

        # run initialization
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(self.global_step, 0))

        # training loop variables
        checkpoint = 0

        batch_size = self.config.batch_size
        nr_batches_train = int(x.shape[0] / batch_size)

        print('Start training...')
        # EPOCHS
        for epoch in range(max_epoch):
            print('---------- EPOCH %s ----------' % epoch)

            begin = time.time()

            # construct randomly shuffled batches
            trainx = sklearn.utils.shuffle(x)
            trainx_copy = sklearn.utils.shuffle(x)

            train_loss_dis_xz, train_loss_dis_xx, train_loss_dis_zz, \
            train_loss_dis, train_loss_gen, train_loss_enc = [0, 0, 0, 0, 0, 0]

            # fit one batch
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # train discriminator
                feed_dict = {self.x_pl: trainx[ran_from:ran_to],
                             self.z_pl: np.random.normal(size=[batch_size, self.config.latent_dim]),
                             self.is_training_pl: True,
                             self.learning_rate: self.config.learning_rate}

                _, _, _, ld, ldxz, ldxx, ldzz, step = sess.run([self.train_dis_op_xz,
                                                                self.train_dis_op_xx,
                                                                self.train_dis_op_zz,
                                                                self.loss_discriminator,
                                                                self.dis_loss_xz,
                                                                self.dis_loss_xx,
                                                                self.dis_loss_zz,
                                                                self.global_step],
                                                               feed_dict=feed_dict)
                train_loss_dis += ld
                train_loss_dis_xz += ldxz
                train_loss_dis_xx += ldxx
                train_loss_dis_zz += ldzz

                # train generator and encoder
                feed_dict = {self.x_pl: trainx_copy[ran_from:ran_to],
                             self.z_pl: np.random.normal(size=[batch_size, self.config.latent_dim]),
                             self.is_training_pl: True,
                             self.learning_rate: self.config.learning_rate}
                _, _, le, lg = sess.run([self.train_gen_op,
                                         self.train_enc_op,
                                         self.loss_encoder,
                                         self.loss_generator],
                                        feed_dict=feed_dict)
                train_loss_gen += lg
                train_loss_enc += le

                # end of batch

                if self.config.enable_sm:
                    sm = sess.run(self.sum_op, feed_dict=feed_dict)
                    writer.add_summary(sm, step)

                if step % self.config.checkpoint_freq != 0: continue
                # checkpoint stuff:


                print('saving checkpoint %s' % checkpoint)
                saver.save(sess, logdir + '/model', global_step=checkpoint)

                if evaluator is not None:
                    print('evaluating checkpoint %s' % checkpoint)
                    evaluator.evaluate(self, checkpoint, {})
                    evaluator.save_results(logdir)

                checkpoint += 1

            # end of epoch
            train_loss_gen /= nr_batches_train
            train_loss_enc /= nr_batches_train
            train_loss_dis /= nr_batches_train
            train_loss_dis_xz /= nr_batches_train
            train_loss_dis_xx /= nr_batches_train
            train_loss_dis_zz /= nr_batches_train

            if self.config.allow_zz:
                print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | "
                      "loss dis = %.4f | loss dis xz = %.4f | loss dis xx = %.4f | "
                      "loss dis zz = %.4f"
                      % (epoch, time.time() - begin, train_loss_gen,
                         train_loss_enc, train_loss_dis, train_loss_dis_xz,
                         train_loss_dis_xx, train_loss_dis_zz))
            else:
                print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | "
                      "loss dis = %.4f | loss dis xz = %.4f | loss dis xx = %.4f | "
                      % (epoch, time.time() - begin, train_loss_gen,
                         train_loss_enc, train_loss_dis, train_loss_dis_xz,
                         train_loss_dis_xx))

    def load(self, file):
        saver = tf.train.Saver()
        saver.restore(self.sess, file)

    def save(self, path):
        pass


def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter


def display_parameters(batch_size, starting_lr, ema_decay, degree, label,
                       allow_zz, score_method, do_spectral_norm):
    """See parameters
    """
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Degree for L norms: ', degree)
    print('Anomalous label: ', label)
    print('Score method: ', score_method)
    print('Discriminator zz enabled: ', allow_zz)
    print('Spectral Norm enabled: ', do_spectral_norm)


def display_progression_epoch(j, id_max):
    """See epoch progression
    """
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


def create_logdir(dataset, label, rd,
                  allow_zz, score_method, do_spectral_norm):
    """ Directory to save training logs, weights, biases, etc."""
    model = 'alad_sn{}_dzz{}'.format(do_spectral_norm, allow_zz)
    return "train_logs/{}/{}/dzzenabled{}/{}/label{}/" \
           "rd{}".format(dataset, model, allow_zz,
                         score_method, label, rd)
