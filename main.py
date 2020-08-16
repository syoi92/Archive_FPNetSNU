import argparse
import os
import tensorflow as tf
tf.set_random_seed(19)
from model import FPNet

parser = argparse.ArgumentParser(description='')
# setting
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=1, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=550, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=512, help='then crop to this size')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=6, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.00001, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--overlapping', dest='overlapping', type=float, default=0.3, help='overlapping when using patch training')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000,
                    help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100,
                    help='print the debug information every print_freq iterations')
# train status
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False,
                    help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--use_patch', dest='use_patch', type=bool, default=True, help='True when using patch training')
# file paths
parser.add_argument('--dataset_dir', dest='dataset_dir', default='ReScaled_2', help='path of the dataset')
parser.add_argument('--model_id', dest='model_id', default='0000', help='serial number of a model')
# controllable
parser.add_argument('--which_net', dest='which_net', default='resnet51', help='resnet51 or resnet152')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--scale', dest='scale', type= float, default=1, help='scale ratio from 2 mode_gap')
args = parser.parse_args()

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

def main(_):
    if not os.path.join('./checkpoints', args.model_id):
        os.makedirs('./checkpoints', args.model_id)
    if not os.path.join('./samples', args.model_id):
        os.makedirs('./samples', args.model_id)
    if not os.path.join('./logs', args.model_id):
        os.makedirs('./logs', args.model_id)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    if not args.phase == 'train': # when test phase, automatically setting the configuration of the model
        model_folder = os.listdir(os.path.join('./checkpoints', args.model_id)
        model_config = model_folder[0].split('_')
        args.dataset_dir = model_config[1]
        args.dataset_dir = model_config[1]
        args.fine_size = model_config[2]
        args.scale = model_config[3]
        args.which_net = model_config[4]
        args.ngf = model_config[5]

        if not os.path.join('./tests', args.model_id):
           os.makedirs('./tests', args.model_id)

        './datasets/{}/*.*'.format(self.dataset_dir + '/trainA')
        
        print('Model Configuration\n\
            model_id: {}\ndataset_dir = {}\nimg_size = {}\n\
                scale = {}\nwhich_net = {}\n# of first filter = {}\n'.format(args.model_id, args.dataset_dir,\ args.fine_size, args.scale, args.which_net, args.ngf))

    with tf.Session(config=tfconfig) as sess:
        model = FPNet(sess, args)
        if args.phase == 'train':
            model.train(args)
        else:
            model.patch_test(args) if args.use_patch else model.test(args)

if __name__ == '__main__':
    tf.app.run()
