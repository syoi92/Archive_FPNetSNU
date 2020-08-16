import argparse
import tensorflow as tf
import cv2
from collections import Counter
import scipy.sparse as sps

from glob import glob
import os
import time
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_path', dest='src_path', default='./sample.png', help='')
parser.add_argument('--model_dir', dest='model_dir', default='./model', help='')
parser.add_argument('--output_dir', dest='output_dir', default='./output', help='')
parser.add_argument('--use_scaleAdjust', dest='use_scaleAdjust', type=bool, default=False, help='')
parser.add_argument('--use_gridTest', dest='use_gridTest', type=bool, default=True, help='')
args = parser.parse_args()


def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # parameters
    pivot_gap = 2  # Pivot pixel gap that will be used for scale adjusting
    img_size = 512  # input img size of DL; (img_size x img_size)
    overlapping = 0.1

    src = cv2.imread(args.src_path, cv2.IMREAD_GRAYSCALE)
    # scaleAdjust
    if args.use_scaleAdjust:
        cur_mode_gap = search_mode_gap(src)  # In a floor plan, calculating a white pixel gap that is the mode
        scale_ratio = pivot_gap / cur_mode_gap
        if not scale_ratio == 1:
            src = cv2.resize(src, (0, 0), fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA)

    # gridTest
    if not args.use_gridTest:
        n, m = src.shape
        if abs(n - m) > 0.05 * max(n, m):  # padding to the square shape only when w-h ratio is far from 1
            if n > m:
                pad = 255 * np.ones((n, int((n - m) / 2)))
                src = np.hstack((pad, src, pad))
            else:
                pad = 255 * np.ones((int((m - n) / 2), m))
                src = np.vstack((pad, src, pad))
        src = cv2.resize(src, (img_size, img_size), interpolation=cv2.INTER_AREA)  # resizing to DL's input size

    # Deep Learning
    model_meta = glob(os.path.join(args.model_dir, '*.meta'))
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        model_saver = tf.train.import_meta_graph(model_meta[0])
        model_saver.restore(sess, os.path.splitext(model_meta[0])[0])
        print("Model Restored- %s\ntime: 0.00s" % os.path.splitext(model_meta[0])[0])
        start_time = time.time()
        if not len(model_meta) == 1:
            print("One model is not specified in %s" % args.model_dir)

        g = tf.get_default_graph()
        test_img = g.get_tensor_by_name("test_img:0")
        test_logit = tf.get_collection('test_logit')[0]

        if not args.use_gridTest:
            src = [load_test_data(src)]
            src = np.array(src).astype(np.float32)
            logit = sess.run(test_logit, feed_dict={test_img: src})
            save_logit(logit, os.path.join(args.output_dir, os.path.basename(args.src_path)))
            print("Prediction stored at %s" % args.output_dir)
            print("Time: %5.2fs" % (time.time()-start_time))

        else:
            # when using girdTest, just padding when src size is smaller than (img_size, img_size)
            pad_h, pad_w = -1, -1
            n, m = src.shape
            if n < img_size:
                pad_h = img_size - n
                src = np.vstack((src, 255 * np.ones((pad_h, m))))
            n, m = src.shape
            if m < img_size:
                pad_w = img_size - m
                src = np.hstack((src, 255 * np.ones((n, pad_w))))

            height, width = src.shape
            print("height: %s, width: %s" % (height, width))
            logit = np.zeros([1, height, width, 6])
            w_num = int(width / img_size / (1 - overlapping))
            h_num = int(height / img_size / (1 - overlapping))
            print("w_num: %s, h_num: %s" % (w_num, h_num))

            for (h, w) in [(h_, w_) for h_ in range(h_num) for w_ in range(w_num)]:
                h1 = int(h / (h_num - 1) * (height - img_size)) if not h_num == 1 else 0
                w1 = int(w / (w_num - 1) * (width - img_size)) if not w_num == 1 else 0
                cur_img = src[h1:h1 + img_size, w1:w1 + img_size]
                cur_img = [load_test_data(cur_img)]
                cur_img = np.array(cur_img).astype(np.float32)
                cur_logit = sess.run(test_logit, feed_dict={test_img: cur_img})
                logit[0, h1:h1 + img_size, w1:w1 + img_size] += cur_logit[0]
                print(("Path processing... [%3d/%3d]" % (h*w_num + w + 1, w_num * h_num)))


            if pad_h > 0:
                logit = logit[:, :-pad_h, :, :]
            if pad_w > 0:
                logit = logit[:, :, :-pad_w, :]
            save_logit(logit, os.path.join(args.output_dir, os.path.basename(args.src_path)))
            print("Prediction stored at %s" % args.output_dir)
            print("Time: %5.2fs" % (time.time() - start_time))



def search_mode_gap(src):
    n, m = src.shape
    src_centercropped = src[int(n * 0.25):int(n * 0.75), int(m * 0.25):int(m * 0.75)]
    bi_src = (src_centercropped < 250) * 1.0

    def gap_counts(sps_src, pixel_gaps):
        for it in range(len(sps_src.indptr) - 1):
            for jt in range(sps_src.indptr[it], sps_src.indptr[it + 1] - 1):
                gg = sps_src.indices[jt + 1] - sps_src.indices[jt] - 1
                if gg > 0:
                    pixel_gaps.append(gg)
        return pixel_gaps

    gaps = []
    tmp = sps.csr_matrix(bi_src)
    gap_counts(tmp, gaps)
    tmp = sps.csc_matrix(bi_src)
    gap_counts(tmp, gaps)

    g_counts = Counter(gaps)
    mode_gap = max(g_counts.keys(), key=(lambda k: g_counts[k]))
    return mode_gap


def load_test_data(src):
    n, m = src.shape
    test_img = src / 127.5 - 1
    test_img = np.array(test_img).reshape([n, m, 1])
    return test_img


def save_logit(logits, image_path):
    def target_label2rgb(target_np, label_colors):
        width, height = target_np.shape
        target_img = np.zeros([width, height, 3], dtype=np.uint8)
        target_img[:] = label_colors[0]  # background
        for it in range(5):
            rr, cc = np.where(target_np == it + 1)
            target_img[rr, cc, :] = label_colors[it + 1]
        return target_img

    ColorMap = np.array([[0, 0, 255], [255, 232, 131], [255, 0, 0],
                         [0, 255, 0], [131, 89, 161], [175, 89, 62]])
    logit = logits[0]
    pred_label = np.argmax(logit, axis=-1)
    img = target_label2rgb(pred_label, ColorMap)
    return cv2.imwrite(image_path, img[:, :, ::-1])


if __name__ == '__main__':
    main()
