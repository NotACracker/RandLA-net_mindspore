import datetime
import os
import time
import argparse
import pickle
import shutil
from pathlib import Path


import mindspore as ms
from mindspore import Model, Tensor, context, load_checkpoint, load_param_into_net, nn, ops
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn import Adam
from mindspore.profiler.profiling import Profiler
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import SummaryCollector
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager


from src.data.dataset import dataloader, ms_map
from src.utils.tools import ConfigS3DIS as cfg
from src.utils.logger import get_logger
from src.model.model import RandLANet, RandLAWithLoss, TrainingWrapper, get_param_groups


def prepare_network(weights, cfg, args):
    """Prepare Network"""

    d_in = 6
    bias = True if args.device_target=='GPU' else False
    network = RandLANet(d_in, cfg.num_classes, bias=bias)
    network = RandLAWithLoss(network, weights, cfg.num_classes)

    return network


def train(args):

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

    logger = get_logger(args.outputs_dir, args.rank)
    
    for arg in vars(args):
      logger.info('%s: %s' %(arg, getattr(args, arg)))

    logger.info('Computing weights...')

    n_samples = Tensor(cfg.class_weights, ms.float32)
    ratio_samples = n_samples / ops.ReduceSum()(n_samples)
    weights = 1 / (ratio_samples + 0.02)
    weights.expand_dims(axis=0)

    logger.info('Done')

    network = prepare_network(weights, cfg, args)

    decay_lr = nn.ExponentialDecayLR(cfg.learning_rate, cfg.lr_decays, decay_steps=cfg.train_steps, is_stair=True)
    opt = Adam(
        params = get_param_groups(network),
        learning_rate = decay_lr,
        loss_scale = cfg.loss_scale
    )

    log = {'cur_epoch':1,'cur_step':1,'best_epoch':1,'besr_miou':0.0}
    if not os.path.exists(args.outputs_dir + '/log.pkl'):
        f = open(args.outputs_dir + '/log.pkl', 'wb')
        pickle.dump(log, f)
        f.close()

    #data loader
    train_loader, val_loader, dataset = dataloader(
        cfg.dataset,
        args,
        num_parallel_workers=8,
        shuffle=False
    )

    train_loader = train_loader.batch(batch_size = args.batch_size,
                                      per_batch_map=ms_map,
                                      input_columns=["xyz","colors","labels","q_idx","c_idx"],
                                      output_columns=["features","labels","input_inds","cloud_inds",
                                                  "p0","p1","p2","p3","p4",
                                                  "n0","n1","n2","n3","n4",
                                                  "pl0","pl1","pl2","pl3","pl4",
                                                  "u0","u1","u2","u3","u4"],
                                      drop_remainder=True)
    
    begin_epoch = log['cur_epoch']
    logger.info('==========begin training===============')
    
    #loss scale manager
    loss_scale_manager = DynamicLossScaleManager() if args.scale else None

    amp_level = 'O0' if args.device_target=='GPU' else 'O3'
    if args.scale:
      model = Model(network,
                    amp_level=amp_level,
                    keep_batchnorm_fp32=True,
                    loss_scale_manager=loss_scale_manager,
                    loss_fn=None,
                    optimizer=opt)
    else:
      model = Model(network,
                    amp_level=amp_level,
                    keep_batchnorm_fp32=True,
                    loss_fn=None,
                    optimizer=opt)

    # callback for loss & time cost
    loss_cb = LossMonitor(50)
    time_cb = TimeMonitor(data_size=cfg.train_steps)
    cbs = [loss_cb, time_cb]

    # callback for saving ckpt
    config_ckpt = CheckpointConfig(save_checkpoint_steps= cfg.train_steps, keep_checkpoint_max=70)
    ckpt_cb = ModelCheckpoint(prefix='randla', directory=os.path.join(args.outputs_dir,'ckpt'), config=config_ckpt)
    cbs += [ckpt_cb]
    
    #summary collector
    summary_collector = SummaryCollector(summary_dir=os.path.join(args.outputs_dir, 'summary'))
    cbs += [summary_collector]

    model.train(args.epochs,
                train_loader,
                callbacks=cbs,
                dataset_sink_mode=False)

    logger.info('==========end training===============')


if __name__ == "__main__":
    """Parse program arguments"""
    parser = argparse.ArgumentParser(
        prog='RandLA-Net',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')

    expr.add_argument('--epochs', type=int, help='max epochs',
                        default=100)

    expr.add_argument('--batch_size', type=int, help='batch size',
                        default=6)

    expr.add_argument('--val_area', type=str, help='area to validate',
                        default='Area_5')
    
    expr.add_argument('--scale', action='store_true', help='scale or not',
                        default=False)

    dirs.add_argument('--outputs_dir', type=str, help='model to save',
                        default='./runs')

    misc.add_argument('--device_target', type=str, help='Ascend or GPU',
                        default='GPU')
    
    misc.add_argument('--device_id', type=int, help='Ascend/GPU id to use',
                        default=0)

    misc.add_argument('--rank', type=int, help='rank',
                        default=0)

    misc.add_argument('--name', type=str, help='name of the experiment',
                        default=None)

    args = parser.parse_args()

    if args.name is None:
        if args.resume:
            args.name = Path(args.resume).split('/')[-1]
        else:
            args.name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    
    args.outputs_dir = os.path.join(args.outputs_dir, args.name)
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    
    #copy file
    shutil.copy('src/utils/tools.py',str(args.outputs_dir))
    shutil.copy('src/mdoel/model.py',str(args.outputs_dir))
    shutil.copy('train.py',str(args.outputs_dir))
    
    # start train
    train(args)






    




