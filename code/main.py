import os
import argparse
import time
import json
import torch
import logging
import numpy as np
from data import Data_loader
from trainer import Trainer
from agent import Agent


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Experiment setup")

    # Path_dir
    parser.add_argument("--config_dir", default="../config", type=str)
    parser.add_argument('--log_dir', default="../logs", type=str)
    parser.add_argument("--data_dir", default="../datasets", type=str)
    parser.add_argument("--model_dir", default="../saved_models", type=str)

    # Dataset
    parser.add_argument('--dataset', default="WN18RR", type=str)
    parser.add_argument('--num_entity', default=0, type=int, help='DO NOT MANUALLY SET')
    parser.add_argument('--num_relation', default=0, type=int, help='DO NOT MANUALLY SET')

    # Agent configuration
    parser.add_argument('--state_embed_size', default=50, type=int)
    parser.add_argument('--relation_embed_size', default=50, type=int)
    parser.add_argument('--mlp_hidden_size', default=100, type=int, help='')
    parser.add_argument('--use_entity_embed', default=False, type=bool, help='')
    parser.add_argument('--entity_embed_size', default=50, type=int)
    parser.add_argument('--grad_clip_norm', default=5, type=int, help='')
    parser.add_argument('--action_embed_size', default=50, type=int)

    parser.add_argument('--train_times', default=20, type=int, help='??')
    parser.add_argument('--test_times', default=100, type=int, help='')
    parser.add_argument("--train_batch", default=20, type=int, help='Train 200 batch')
    parser.add_argument('--max_out', default=100, type=int, help='Maximum output of each entity')
    parser.add_argument('--max_step_length', default=3, type=int, help='Maximum resaoning path length')

    # Parameters configuration
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--decay_weight', default=0.02, type=float, help='')
    parser.add_argument('--decay_batch', default=100, type=int, help='Decay times')
    parser.add_argument('--decay_rate', default=0.9, type=float)

    parser.add_argument('--gamma', default=1, type=float, help='Rewards discount parameter')
    parser.add_argument('--Lambda', default=0.05, type=float, help='Baseline update_rate')
    parser.add_argument('--beta', default=0.05, type=float, help='Initial decaying_beta')

    # other configuration
    parser.add_argument('--use_cuda', default=True, type=bool)

    return parser.parse_args(args)


def set_logger(args):
    '''
    Write logs to save_path and console
    '''
    log_file = os.path.join(args.log_dir, args.dataset + '.log')
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8') as f:
            pass

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def save_config(args):
    config_file = os.path.join(args.config_dir, args.dataset + ".json")
    argparse_dict = vars(args)
    # with open(config_file, 'w') as fjson:
    #     json.dump(argparse_dict, fjson)
    with open(config_file, "a", encoding='UTF-8') as f:
        f.write(time.strftime("%y-%m-%d %H:%M:%S") + '\n')
        for key, value in sorted(argparse_dict.items(), key=lambda x: x[0]):
            f.write("{}, {}\n".format(key, str(value)))
        f.write('\n')

def main(args):
    if args.use_entity_embed is True:
        args.action_embed_size = args.relation_embed_size + args.entity_embed_size

    set_logger(args)
    logging.info('-------------------SSSSSStart!---------------------')

    data_loader = Data_loader(args)
    args.num_entity = data_loader.num_entity
    args.num_relation = data_loader.num_relation
    save_config(args)

    logging.info('Configuration : %s'% (str(vars(args))))

    agent = Agent(args, data_loader)
    # trainer = Trainer(args, agent, data_loader, plotter)
    trainer = Trainer(args, agent, data_loader)

    trainer.train()
    trainer.save_model()
    trainer.load_model()
    trainer.test()

    logging.info('---------------EEEEEnd!---------------')

if __name__ == "__main__":
    # global plotter
    # plotter = utils.VisdomLinePlotter(env_name='Tutorial Plots')
    # main(parse_args(), plotter)

    main(parse_args())



