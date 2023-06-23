import optuna
import argparse, collections
from omegaconf import OmegaConf
from optuna.integration.wandb import WeightsAndBiasesCallback

import trainer.train_loop as train_loop
from parse_config import ConfigParser
from configuration import CFG
from utils.helper import check_library, all_type_seed
from utils import sync_config


check_library(True)
all_type_seed(CFG, True)


def main(config_path: str, cfg) -> None:
    sync_config(OmegaConf.load(config_path))  # load json config
    # cfg = OmegaConf.structured(CFG)
    # OmegaConf.merge(cfg)  # merge with cli_options
    if cfg.optuna:
        """ Optuna Hyperparameter Optimization """
        study = optuna.create_study(
            study_name=cfg.name,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            storage=f'sqlite:///{cfg.checkpoint_dir}{cfg.name}.db',
            load_if_exists=True
        )
    else:
        getattr(train_loop, cfg.loop)(cfg)  # init object


if __name__ == '__main__':
    # args = argparse.ArgumentParser(description='PyTorch Template')
    # args.add_argument('-c', '--config', default=None, type=str,
    #                   help='config file path (default: None)')
    # args.add_argument('-r', '--resume', default=None, type=str,
    #                   help='path to latest checkpoint (default: None)')
    # args.add_argument('-d', '--device', default=None, type=str,
    #                   help='indices of GPUs to enable (default: all)')
    #
    # # custom cli options to modify configuration from default values given in json file.
    # CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # options = [
    #     CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
    #     CustomArgs(['--bs', '--batch_size'], type=int, target='dataset_class;args;batch_size')
    # ]
    # cli_config = ConfigParser.from_args(args, options)
    main('fbp3_config.json', CFG)
