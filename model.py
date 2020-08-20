from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from collections import namedtuple
from glob import glob

from module import *
from utils import *


class FPNet(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.load_size = args.load_size
        self.input_nc = args.input_nc
        self.output_nc = args.output_nc
        self.label_colors = np.array([[0, 0, 255], [255, 232, 131], [255, 0, 0],
                                      [0, 255, 0], [131, 89, 161], [175, 89, 62]])
        self.dataset_dir = args.dataset_dir
        self.overlapping = args.overlapping
        self.scale = args.scale
        self.is_test = True if args.phase == 'test' else False
        self.use_patch = True if args.use_patch == 1 else False
        self.use_styleloss = 1 if args.use_styleloss else 0

        if not self.use_patch:
            self.load_train_data = load_train_data
        else:
            self.load_train_data = load_train_patch

        if args.which_net == 'resnet152':
            self.resnet = resnet152
        else:
            self.resnet = resnet51

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size load_size scale\
                              gf_dim output_nc overlapping label_colors is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size, args.load_size, args.scale,
                                      args.ngf, args.output_nc, args.overlapping, self.label_colors,
                                      args.phase == 'train'))

        model_dir = "GDIsNfDaSc_%s_%s_%s_%s_%s_%s" % \
                    (args.which_net, self.use_styleloss, self.image_size, args.ngf, self.dataset_dir, self.scale)
        self.sample_model_dir = os.path.join('./samples', args.model_id, model_dir)
        if not os.path.exists(self.sample_model_dir):
            os.makedirs(self.sample_model_dir)
        self.checkpoint_model_dir = os.path.join('./checkpoints', args.model_id, model_dir)
        if not os.path.exists(self.checkpoint_model_dir):
            os.makedirs(self.checkpoint_model_dir)
        self.log_model_dir = os.path.join('./logs', args.model_id)
        if not os.path.exists(self.log_model_dir):
            os.makedirs(self.log_model_dir)
        self.test_dir = os.path.join('./tests', args.model_id)

        self._build_model()
        self.saver = tf.train.Saver() 


    def _build_model(self):
        self.datasets = tf.placeholder(tf.float32,
                                       [None, self.image_size, self.image_size,
                                        self.input_nc + self.output_nc],
                                       name='data_ori')
        self.inputs = self.datasets[:, :, :, :self.input_nc]
        self.labels = self.datasets[:, :, :, self.input_nc:self.input_nc + self.output_nc]

        self.logits = self.resnet(self.inputs, self.options, False, name="resnet")
        self.loss = sce_criterion(self.logits, self.labels)

        # summary
        self.loss_sum = tf.summary.scalar("sce_loss", self.loss)

        # -------- For Test ---------
        self.test_img = tf.placeholder(tf.float32,
                                       [None, self.image_size, self.image_size,
                                        self.input_nc], name='test_img')
        self.test_logit = self.resnet(self.test_img, self.options, True, name="resnet")
        tf.add_to_collection('test_logit', self.test_logit)

        t_vars = tf.trainable_variables()
        #for var in t_vars:
        #    print(var.name)
        self.resnet_vars = [var for var in t_vars if 'resnet' in var.name]

    def train(self, args):
        """Train FPNet"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.loss, var_list=self.resnet_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.log_model_dir, self.sess.graph)

        counter = 1
        start_time = time.time()
        
        if args.continue_train:
            if self.load(self.checkpoint_model_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))

            if not len(dataA) == len(dataB):
                print('Something wrong. # of img and label are not same')
            dataAB = list(zip(dataA, dataB))
            np.random.shuffle(dataAB)
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            if not self.use_patch:
                batch_idxs = min(len(dataAB), args.train_size) // self.batch_size
                for idx in range(0, batch_idxs):
                    batch_files = dataAB[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_images = [self.load_train_data(batch_file, self.options) for batch_file in batch_files]
                    batch_images = np.array(batch_images).astype(np.float32)

                    _, cur_loss, summary_str = self.sess.run(
                        [self.optim, self.loss, self.loss_sum],
                        feed_dict={self.datasets: batch_images,
                                   self.lr: lr})
                    self.writer.add_summary(summary_str, counter)

                    counter += 1
                    cur_file = os.path.basename(dataAB[idx][0])
                    print(("Epoch: [%2d] [%3d/%3d] - %s | time: %4.4f loss: %4.4f" % (
                        epoch, idx, batch_idxs, cur_file, time.time() - start_time, cur_loss)))

                    if np.mod(counter, args.print_freq) == 1:
                        self.sample_model(self.sample_model_dir, epoch, idx)
                    if np.mod(counter, args.save_freq) == 2:
                        self.save(self.checkpoint_model_dir, counter)
            else:
                idxs = len(dataAB)
                for idx in range(0, idxs):
                    _, crop_img, crop_label, crop_white = self.load_train_data(dataAB[idx], self.options)
                    crop_data = np.concatenate((crop_img, crop_label), axis=-1)
                    crop_data_ = crop_data[crop_white == False]
                    pidxs = np.sum(crop_white == False)
                    batch_pidxs = pidxs // self.batch_size
                    cur_file = os.path.basename(dataAB[idx][0])

                    for pidx in range(0, batch_pidxs):
                        batch_images = crop_data_[pidx * self.batch_size: (pidx + 1) * self.batch_size]
                        batch_images = np.array(batch_images).astype(np.float32)

                        _, cur_loss, summary_str = self.sess.run(
                            [self.optim, self.loss, self.loss_sum],
                            feed_dict={self.datasets: batch_images,
                                       self.lr: lr})
                        self.writer.add_summary(summary_str, counter)

                        counter += 1
                        print(("Epoch: [%2d] [%3d/%3d] - %s | Patch: [%4d/%4d] time: %4.4f loss: %4.4f" % (
                            epoch, idx, idxs, cur_file, pidx + 1, batch_pidxs, time.time() - start_time, cur_loss)))

                        if np.mod(counter, args.print_freq) == 1:
                            self.patch_sample_model(self.sample_model_dir, epoch, idx)
                        if np.mod(counter, args.save_freq) == 2:
                            self.save(self.checkpoint_model_dir, counter)


    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
        np.random.shuffle(dataA)
        batch_files = dataA[:self.batch_size]
        sample_images = [load_test_data(batch_file, self.image_size) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        logits = self.sess.run(
            self.test_logit,
            feed_dict={self.test_img: sample_images})
        save_pred(logits, './{}/B_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx), self.label_colors)

    def patch_sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
        np.random.shuffle(dataA)
        img = cv2.imread(dataA[0], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        pad_h, pad_w = -1, -1
        height, width = img.shape[0:2]
        if height < self.image_size:
            pad_h = self.image_size - height
            img = np.vstack((img, 255 * np.ones((pad_h, width))))
        height, width = img.shape[0:2]
        if width < self.image_size:
            pad_w = self.image_size - width
            img = np.hstack((img, 255 * np.ones((height, pad_w))))

        height, width = img.shape[0:2]
        img = np.array(img).reshape([height, width, 1])
        img = (img / 127.5 - 1).astype(np.float32)
        logits = np.zeros([1, height, width, self.output_nc])
        w_num = int(width / self.image_size / (1 - max(self.overlapping, 0.1)))
        h_num = int(height / self.image_size / (1 - max(self.overlapping, 0.1)))
        for (h, w) in [(h_, w_) for h_ in range(h_num) for w_ in range(w_num)]:
            h1 = int(h / (h_num - 1) * (height - self.image_size)) if not h_num == 1 else 0
            w1 = int(w / (w_num - 1) * (width - self.image_size)) if not w_num == 1 else 0
            cur_img = [img[h1:h1 + self.image_size, w1:w1 + self.image_size, :]]
            cur_logit = self.sess.run(self.test_logit, feed_dict={self.test_img: np.array(cur_img)})
            logits[0, h1:h1 + self.image_size, w1:w1 + self.image_size] += cur_logit[0]

        if pad_h > 0:
            logits = logits[:, :-pad_h, :, :]
        if pad_w > 0:
            logits = logits[:, :, :-pad_w, :]

        filename=os.path.basename(dataA[0])
        print(("Save Sample  - %s " % (filename)))
        save_pred(logits, './{}/B_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx), self.label_colors)

    def save(self, checkpoint_dir, step):
        model_name = "FPNet.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test """
        start_test_time = time.clock()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        if self.load(self.checkpoint_model_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.test_dir, 'index.html')
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(self.test_dir,
                                      os.path.basename(sample_file))
            logits = self.sess.run(self.test_logit, feed_dict={self.test_img: sample_image})
            np.save(image_path +'.npy', logits)
            save_pred(logits, image_path, self.label_colors)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                    '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
        
        end_test_time = time.clock()
        print('Running time : ', end_test_time - start_test_time)

    def patch_test(self, args):
        """Patch Test """
        start_test_time = time.clock()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        if self.load(self.checkpoint_model_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.test_dir, 'index.html')
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        idx, idxs = 1, len(sample_files)
        for sample_file in sample_files:
            print('Test image: ' + sample_file)
            img = cv2.imread(sample_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

            pad_h, pad_w = -1, -1
            height, width = img.shape[0:2]
            if height < self.image_size:
                pad_h = self.image_size - height
                img = np.vstack((img, 255 * np.ones((pad_h, width))))
            height, width = img.shape[0:2]
            if width < self.image_size:
                pad_w = self.image_size - width
                img = np.hstack((img, 255 * np.ones((height, pad_w))))

            height, width = img.shape[0:2]
            img = np.array(img).reshape([height, width, 1])
            img = (img / 127.5 -1).astype(np.float32)
            logits = np.zeros([1, height, width, self.output_nc])
            w_num = int(width / self.image_size / (1 - max(self.overlapping, 0.1)))
            h_num = int(height / self.image_size / (1 - max(self.overlapping, 0.1)))
            for (h, w) in [(h_, w_) for h_ in range(h_num) for w_ in range(w_num)]:
                h1 = int(h / (h_num - 1) * (height - self.image_size)) if not h_num == 1 else 0
                w1 = int(w / (w_num - 1) * (width - self.image_size)) if not w_num == 1 else 0
                cur_img = [img[h1:h1 + self.image_size, w1:w1 + self.image_size, :]]
                cur_logit = self.sess.run(self.test_logit, feed_dict={self.test_img: cur_img})
                logits[0, h1:h1 + self.image_size, w1:w1 + self.image_size] += cur_logit[0]
                print(("Test[%2d|%2d]| Path processing... [%3d/%3d]" % (idx, idxs, h*w_num + w + 1, w_num * h_num)))

            idx += 1
            if pad_h > 0:
                logits = logits[:, :-pad_h, :, :]
            if pad_w > 0:
                logits = logits[:, :, :-pad_w, :]

            image_path = os.path.join(self.test_dir, os.path.basename(sample_file))
            save_pred(logits, image_path, self.label_colors)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                    '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()

        end_test_time = time.clock()
        print('Running time : ', end_test_time - start_test_time)


class FPNet_StyleLoss(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.load_size = args.load_size
        self.input_nc = args.input_nc
        self.output_nc = args.output_nc
        self.label_colors = np.array([[0, 0, 255], [255, 232, 131], [255, 0, 0],
                                      [0, 255, 0], [131, 89, 161], [175, 89, 62]])
        self.dataset_dir = args.dataset_dir
        self.overlapping = args.overlapping
        self.scale = args.scale
        self.is_test = True if args.phase == 'test' else False
        self.use_patch = True if args.use_patch == 1 else False
        self.use_styleloss = 1 if not args.use_styleloss == 0 else 0

        if not self.use_patch:
            self.load_train_data = load_train_data
        else:
            self.load_train_data = load_train_patch

        if args.which_net == 'resnet152':
            self.resnet = resnet152
        else:
            self.resnet = resnet51
        self.discriminator = discriminator
        self.style_ratio = args.style_ratio

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size load_size scale\
                              gf_dim df_dim output_nc overlapping label_colors is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size, args.load_size, args.scale,
                                      args.ngf, args.ndf, args.output_nc, args.overlapping, self.label_colors,
                                      args.phase == 'train'))

        model_dir = "GDIsNfDaSc_%s_%s_%s_%s_%s_%s" % \
                    (args.which_net, self.use_styleloss, self.image_size, args.ngf, self.dataset_dir, self.scale)
        self.sample_model_dir = os.path.join('./samples', args.model_id, model_dir)
        if not os.path.exists(self.sample_model_dir):
            os.makedirs(self.sample_model_dir)
        self.checkpoint_model_dir = os.path.join('./checkpoints', args.model_id, model_dir)
        if not os.path.exists(self.checkpoint_model_dir):
            os.makedirs(self.checkpoint_model_dir)
        self.log_model_dir = os.path.join('./logs', args.model_id)
        if not os.path.exists(self.log_model_dir):
            os.makedirs(self.log_model_dir)
        self.test_dir = os.path.join('./tests', args.model_id)

        self._build_model()
        self.saver = tf.train.Saver()


    def _build_model(self):
        self.datasets = tf.placeholder(tf.float32,
                                       [None, self.image_size, self.image_size,
                                        self.input_nc + self.output_nc],
                                       name='data_ori')
        self.inputs = self.datasets[:, :, :, :self.input_nc]
        self.labels = self.datasets[:, :, :, self.input_nc:self.input_nc + self.output_nc]
        # build generator
        self.fake_B = self.resnet(self.inputs, self.options, False, name="resnet")
        # build discriminator
        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminator")
        self.DB_real = self.discriminator(self.labels, self.options, reuse=True, name="discriminator")

        # generator_loss
        self.style_loss = sce_criterion(self.DB_fake, tf.ones_like(self.DB_fake))
        self.geo_loss = sce_criterion(self.fake_B, self.labels)
        self.g_loss = self.style_ratio * self.style_loss + self.geo_loss

        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_nc], name='fake_B_sample')
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminator")

        # discriminator_loss
        self.db_loss_real = sce_criterion(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = sce_criterion(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample)) # this part is opposite to generator loss
        self.d_loss = (self.db_loss_real + self.db_loss_fake)/2

        # summary
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.style_loss_sum = tf.summary.scalar("style_loss", self.style_loss)
        self.geo_loss_sum = tf.summary.scalar("geo_loss", self.geo_loss)
        self.g_sum = tf.summary.merge(
            [self.style_loss_sum, self.geo_loss_sum, self.g_loss_sum])

        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.db_loss_real_sum, self.db_loss_fake_sum, self.d_loss_sum])

        # -------- For Test ---------
        self.test_img = tf.placeholder(tf.float32,
                                       [None, self.image_size, self.image_size,
                                        self.input_nc], name='test_img')
        self.test_logit = self.resnet(self.test_img, self.options, True, name="resnet")
        tf.add_to_collection('test_logit', self.test_logit)

        t_vars = tf.trainable_variables()
        #for var in t_vars:
        #    print(var.name)
        self.g_vars = [var for var in t_vars if 'resnet' in var.name]
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]


    def train(self, args):
        """Train FPNet"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.log_model_dir, self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(self.checkpoint_model_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            if not len(dataA) == len(dataB):
                print('Something wrong. # of img and label are not same')
            dataAB = list(zip(dataA, dataB))
            np.random.shuffle(dataAB)
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            if not self.use_patch:
                batch_idxs = min(len(dataAB), args.train_size) // self.batch_size
                for idx in range(0, batch_idxs):
                    batch_files = dataAB[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_images = [self.load_train_data(batch_file, self.options) for batch_file in batch_files]
                    batch_images = np.array(batch_images).astype(np.float32)

                    # Update G network and record fake outputs
                    fake_B, _, cur_style_loss, cur_geo_loss, summary_str = self.sess.run(
                        [self.fake_B, self.g_optim, self.style_loss, self.geo_loss, self.g_sum],
                        feed_dict={self.datasets: batch_images, self.lr: lr})
                    self.writer.add_summary(summary_str, counter)

                    # Update D network and record fake outputs
                    _, summary_str = self.sess.run(
                        [self.d_optim, self.d_sum],
                        feed_dict={self.datasets: batch_images, self.fake_B_sample: fake_B,
                                   self.lr: lr})
                    self.writer.add_summary(summary_str, counter)

                    counter += 1
                    cur_file = os.path.basename(dataAB[idx][0])
                    print(("Epoch: [%2d] [%3d/%3d] - %s | time: %4.4f Loss[Style|Geo]: %4.4f|%4.4f" % (
                        epoch, idx, batch_idxs, cur_file, time.time() - start_time, cur_style_loss, cur_geo_loss)))

                    if np.mod(counter, args.print_freq) == 1:
                        self.sample_model(self.sample_model_dir, epoch, idx)
                    if np.mod(counter, args.save_freq) == 2:
                        self.save(self.checkpoint_model_dir, counter)
            else:
                idxs = len(dataAB)
                for idx in range(0, idxs):
                    _, crop_img, crop_label, crop_white = self.load_train_data(dataAB[idx], self.options)
                    crop_data = np.concatenate((crop_img, crop_label), axis=-1)
                    crop_data_ = crop_data[crop_white == False]
                    pidxs = np.sum(crop_white == False)
                    batch_pidxs = pidxs // self.batch_size
                    cur_file = os.path.basename(dataAB[idx][0])

                    for pidx in range(0, batch_pidxs):
                        batch_images = crop_data_[pidx * self.batch_size: (pidx + 1) * self.batch_size]
                        batch_images = np.array(batch_images).astype(np.float32)

                        # Update G network and record fake outputs
                        fake_B, _, cur_style_loss, cur_geo_loss, summary_str = self.sess.run(
                            [self.fake_B, self.g_optim, self.style_loss, self.geo_loss, self.g_sum],
                            feed_dict={self.datasets: batch_images, self.lr: lr})
                        self.writer.add_summary(summary_str, counter)

                        # Update D network and record fake outputs
                        _, summary_str = self.sess.run(
                            [self.d_optim, self.d_sum],
                            feed_dict={self.datasets: batch_images, self.fake_B_sample: fake_B,
                                       self.lr: lr})
                        self.writer.add_summary(summary_str, counter)

                        counter += 1
                        print(("Epoch:[%2d][%3d/%3d] - %s|Patch:[%4d/%4d] time: %4.4f loss[Style|Geo]: %4.4f|%4.4f" % (
                            epoch, idx, idxs, cur_file, pidx + 1, batch_pidxs, time.time() - start_time, cur_style_loss, cur_geo_loss)))

                        if np.mod(counter, args.print_freq) == 1:
                            self.patch_sample_model(self.sample_model_dir, epoch, idx)
                        if np.mod(counter, args.save_freq) == 2:
                            self.save(self.checkpoint_model_dir, counter)


    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
        np.random.shuffle(dataA)
        batch_files = dataA[:self.batch_size]
        sample_images = [load_test_data(batch_file, self.image_size) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        logits = self.sess.run(
            self.test_logit,
            feed_dict={self.test_img: sample_images})
        save_pred(logits, './{}/B_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx), self.label_colors)

    def patch_sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
        np.random.shuffle(dataA)
        img = cv2.imread(dataA[0], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        pad_h, pad_w = -1, -1
        height, width = img.shape[0:2]
        if height < self.image_size:
            pad_h = self.image_size - height
            img = np.vstack((img, 255 * np.ones((pad_h, width))))
        height, width = img.shape[0:2]
        if width < self.image_size:
            pad_w = self.image_size - width
            img = np.hstack((img, 255 * np.ones((height, pad_w))))

        height, width = img.shape[0:2]
        img = np.array(img).reshape([height, width, 1])
        img = (img / 127.5 - 1).astype(np.float32)
        logits = np.zeros([1, height, width, self.output_nc])
        w_num = int(width / self.image_size / (1 - max(self.overlapping, 0.1)))
        h_num = int(height / self.image_size / (1 - max(self.overlapping, 0.1)))
        for (h, w) in [(h_, w_) for h_ in range(h_num) for w_ in range(w_num)]:
            h1 = int(h / (h_num - 1) * (height - self.image_size)) if not h_num == 1 else 0
            w1 = int(w / (w_num - 1) * (width - self.image_size)) if not w_num == 1 else 0
            cur_img = [img[h1:h1 + self.image_size, w1:w1 + self.image_size, :]]
            cur_logit = self.sess.run(self.test_logit, feed_dict={self.test_img: np.array(cur_img)})
            logits[0, h1:h1 + self.image_size, w1:w1 + self.image_size] += cur_logit[0]

        if pad_h > 0:
            logits = logits[:, :-pad_h, :, :]
        if pad_w > 0:
            logits = logits[:, :, :-pad_w, :]

        filename=os.path.basename(dataA[0])
        print(("Save Sample  - %s " % (filename)))
        save_pred(logits, './{}/B_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx), self.label_colors)

    def save(self, checkpoint_dir, step):
        model_name = "FPNet.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test """
        start_test_time = time.clock()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        if self.load(self.checkpoint_model_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.test_dir, 'index.html')
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(self.test_dir,
                                      os.path.basename(sample_file))
            logits = self.sess.run(self.test_logit, feed_dict={self.test_img: sample_image})
            np.save(image_path +'.npy', logits)
            save_pred(logits, image_path, self.label_colors)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                    '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()

        end_test_time = time.clock()
        print('Running time : ', end_test_time - start_test_time)

    def patch_test(self, args):
        """Patch Test """
        start_test_time = time.clock()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        if self.load(self.checkpoint_model_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.test_dir, 'index.html')
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        idx, idxs = 1, len(sample_files)
        for sample_file in sample_files:
            print('Test image: ' + sample_file)
            img = cv2.imread(sample_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

            pad_h, pad_w = -1, -1
            height, width = img.shape[0:2]
            if height < self.image_size:
                pad_h = self.image_size - height
                img = np.vstack((img, 255 * np.ones((pad_h, width))))
            height, width = img.shape[0:2]
            if width < self.image_size:
                pad_w = self.image_size - width
                img = np.hstack((img, 255 * np.ones((height, pad_w))))

            height, width = img.shape[0:2]
            img = np.array(img).reshape([height, width, 1])
            img = (img / 127.5 -1).astype(np.float32)
            logits = np.zeros([1, height, width, self.output_nc])
            w_num = int(width / self.image_size / (1 - max(self.overlapping, 0.1)))
            h_num = int(height / self.image_size / (1 - max(self.overlapping, 0.1)))
            for (h, w) in [(h_, w_) for h_ in range(h_num) for w_ in range(w_num)]:
                h1 = int(h / (h_num - 1) * (height - self.image_size)) if not h_num == 1 else 0
                w1 = int(w / (w_num - 1) * (width - self.image_size)) if not w_num == 1 else 0
                cur_img = [img[h1:h1 + self.image_size, w1:w1 + self.image_size, :]]
                cur_logit = self.sess.run(self.test_logit, feed_dict={self.test_img: cur_img})
                logits[0, h1:h1 + self.image_size, w1:w1 + self.image_size] += cur_logit[0]
                print(("Test[%2d|%2d]| Path processing... [%3d/%3d]" % (idx, idxs, h*w_num + w + 1, w_num * h_num)))

            idx += 1
            if pad_h > 0:
                logits = logits[:, :-pad_h, :, :]
            if pad_w > 0:
                logits = logits[:, :, :-pad_w, :]

            image_path = os.path.join(self.test_dir, os.path.basename(sample_file))
            save_pred(logits, image_path, self.label_colors)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                    '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()

        end_test_time = time.clock()
        print('Running time : ', end_test_time - start_test_time)