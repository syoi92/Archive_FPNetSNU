from __future__ import division
import numpy as np
import cv2


def load_test_data(image_path, fine_size=512):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (fine_size, fine_size), interpolation=cv2.INTER_AREA)
    img = np.array(img).reshape([fine_size, fine_size, 1]) / 127.5 -1
    return img


def load_train_data(image_path, options):
    img_A = cv2.imread(image_path[0], cv2.IMREAD_GRAYSCALE)
    img_B = cv2.imread(image_path[1], cv2.IMREAD_COLOR)
    img_B = img_B[:, :, ::-1]  # OpenCV read images in BGR order
    label_B = target_rgb2label(img_B, options.label_colors)  # label map

    is_cropping = True if np.random.random() > 0.8 else False
    if not is_cropping:
        img_A = cv2.resize(img_A, (options.load_size, options.load_size), interpolation=cv2.INTER_AREA)
        label_B = cv2.resize(label_B, (options.load_size, options.load_size), interpolation=cv2.INTER_NEAREST)
        h1 = int(np.ceil(np.random.uniform(1e-2, options.load_size - options.image_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, options.load_size - options.image_size)))
        img_A = img_A[h1:h1 + options.image_size, w1:w1 + options.image_size]
        label_B = label_B[h1:h1 + options.image_size, w1:w1 + options.image_size]
    else:
        img_A = cv2.resize(img_A, (options.image_size, options.image_size), interpolation=cv2.INTER_AREA)
        label_B = cv2.resize(label_B, (options.image_size, options.image_size), interpolation=cv2.INTER_NEAREST)

    # img_A: (fine_size, fine_size) >> np_A: (fine_size, fine_size, 1)
    np_A = np.array(img_A) / 127.5 -1
    np_A = np_A.reshape([options.image_size, options.image_size, 1])

    # label_B: (fine_size,fine_size) >> np_B: (fine_size, fine_size, output_nc)
    np_B = np.zeros([options.image_size, options.image_size, options.output_nc], dtype=np.float32)
    for it in range(options.output_nc):
        rr, cc = np.where(label_B == it)
        np_B[rr, cc, it] = 1

    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim )
    return np.concatenate((np_A, np_B), axis=2)


def save_pred(logits, image_path, label_colors):
    logit = logits[0] if len(logits.shape) == 4 else logits
    pred_label = np.argmax(logit, axis=-1)
    img = target_label2rgb(pred_label, label_colors)
    return cv2.imwrite(image_path, img[:, :, ::-1])


def target_rgb2label(target_img, label_colors):
    width, height, _ = target_img.shape
    target_label = np.zeros([width, height], dtype=np.uint8)
    for it in range(1, len(label_colors)):
        rr, cc = np.where(np.all(target_img == label_colors[it], axis=-1))
        target_label[rr, cc] = it
    return target_label


def target_label2rgb(target_np, label_colors):
    width, height = target_np.shape
    target_img = np.zeros([width, height, 3], dtype=np.uint8)
    target_img[:] = label_colors[0]  # background
    for it in range(np.max(target_np)):
        rr, cc = np.where(target_np == it+1)
        target_img[rr, cc, :] = label_colors[it+1]
    return target_img

# grid
def load_train_patch(image_path, options):
    img_A = cv2.imread(image_path[0], cv2.IMREAD_GRAYSCALE)
    img_B = cv2.imread(image_path[1], cv2.IMREAD_COLOR)
    img_B = img_B[:, :, ::-1]  # OpenCV read images in BGR order
    label_B = target_rgb2label(img_B, options.label_colors)  # label map

    #  scale_ratio
    img_A = cv2.resize(img_A, (0,0), fx=options.scale, fy=options.scale, interpolation=cv2.INTER_AREA)
    label_B = cv2.resize(label_B, (0,0), fx=options.scale, fy=options.scale, interpolation=cv2.INTER_NEAREST)

    height, width = img_A.shape[0:2]
    if height < options.image_size:
        pad_h = options.image_size-height
        img_A = np.vstack((img_A, 255 * np.ones((pad_h, width))))
    height, width = img_A.shape[0:2]
    if width < options.image_size:
        pad_w = options.image_size - width
        img_A = np.hstack((img_A, 255 * np.ones((height, pad_w))))

    # patch
    height, width = img_A.shape[0:2]
    crop_name, crop_img, crop_label, crop_white = [], [], [], []
    overlapping = min((1 - 1 / options.image_size), options.overlapping)
    w_num = int(width / options.image_size / (1-overlapping))
    h_num = int(height / options.image_size / (1-overlapping))

    np_A = np.array(img_A).reshape([height, width, 1])
    np_B = np.zeros([height, width, len(options.output_nc)], dtype=np.float32)
    for it in range(options.output_nc):
        rr, cc = np.where(label_B == it)
        np_B[rr, cc, it] = 1

    for (h, w) in [(h_, w_) for h_ in range(h_num) for w_ in range(w_num)]:
        h1 = int(h / (h_num - 1) * (height - options.image_size)) if not h_num == 1 else 0
        w1 = int(w / (w_num - 1) * (width - options.image_size)) if not w_num == 1 else 0
        cur_img = np_A[h1:h1 + options.image_size, w1:w1 + options.image_size, :]
        cur_label = np_B[h1:h1 + options.image_size, w1:w1 + options.image_size, :]
        is_white = True if (np.sum(cur_img > 245) / options.image_size ** 2) > 0.98 else False  # threshold = 0.95

        cur_img = cur_img / 127.5 - 1
        crop_name.append((w, h))
        crop_img.append(cur_img)
        crop_label.append(cur_label)
        crop_white.append(is_white)

    return np.array(crop_name), np.array(crop_img), np.array(crop_label), np.array(crop_white)