import find_mxnet
import mxnet as mx
import argparse
import os, sys
import math
import logging
# import train_model

parser = argparse.ArgumentParser(description='train an image classifer on cifar10')
parser.add_argument('--network', type=str, default='inception-bn-28-small',
                    help = 'the cnn to use')
parser.add_argument('--data-dir', type=str, default='/home/data/cifar10/',
                    help='the input data directory')
parser.add_argument('--gpus', type=str, default='0',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, default=50000,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=.05,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--model-prefix', type=str, 
                    help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str, 
                    help='the prefix of the model to save')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
# save training log into log file to plot training curve
parser.add_argument('--log-dir', type=str, 
                    help = 'the path of log dir')
parser.add_argument('--log-file', type=str, 
                    help = 'the name of log file')
args = parser.parse_args()

# download data if necessary
# def _download(data_dir):
#     if not os.path.isdir(data_dir):
#         os.system("mkdir " + data_dir)
#     os.chdir(data_dir)
#     if (not os.path.exists('train.rec')) or \
#        (not os.path.exists('test.rec')) :
#         os.system("wget http://data.dmlc.ml/mxnet/data/cifar10.zip")
#         os.system("unzip -u cifar10.zip")
#         os.system("mv cifar/* .; rm -rf cifar; rm cifar10.zip")
#     os.chdir("..")

# network
import importlib
net = importlib.import_module('symbol_' + args.network).get_symbol(10)

# data
def get_iterator(args, kv):
    data_shape = (3, 28, 28)
    # if '://' not in args.data_dir:
    #     _download(args.data_dir)

    train = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "train.rec",
        mean_img    = args.data_dir + "mean.bin",
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "test.rec",
        mean_img    = args.data_dir + "mean.bin",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

def get_myiterator(args, kv):
    data_shape = (3, 28, 28)
        
    img_lists = [args.data_dir+"train_lst/cifar10_%d.lst"%i for i in range(10)]
    
    train = mx.io.ImageSampleIter(
        img_lists   = img_lists,
        mean_rgb    = (117,117,117),
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        root        = args.data_dir + 'train/',
        rand_crop   = True,
        rand_mirror = True,
        shuffle     = True)

    val = mx.io.ImageDataIter(
        img_lst     = args.data_dir + "test.lst",
        mean_rgb    = (117,117,117),
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        root        = args.data_dir + 'test/')
    
    return (train, val)


def fit(args, network, data_loader, batch_end_callback=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

    # load model
    model_prefix = args.model_prefix
    # if model_prefix is not None:
    #     model_prefix += "-%d" % (kv.rank)
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}
    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # data
    (train, val) = data_loader(args, kv)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size
    logging.info('Debug: epoch size = %d', epoch_size)

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(epoch_size * args.lr_factor_epoch), 1),
            factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        **model_args)

    eval_metrics = ['accuracy']
    eval_metrics.append(mx.metric.create('ce'))

    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 100))

    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        batch_end_callback = batch_end_callback,
        epoch_end_callback = checkpoint)

# train
# fit(args, net, get_iterator)
fit(args, net, get_myiterator)
