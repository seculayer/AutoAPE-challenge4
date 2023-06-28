import os, warnings
import trainer.train_loop as train_loop
from omegaconf import OmegaConf
from configuration import CFG
from utils.helper import check_library, all_type_seed
from utils import sync_config
from dataset_class.data_preprocessing import add_markdown_token, add_code_token
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ['LRU_CACHE_CAPACITY'] = "1"
check_library(True)
all_type_seed(CFG, True)


def main(config_path: str, cfg) -> None:
    sync_config(OmegaConf.load(config_path))
    add_markdown_token(cfg), add_code_token(cfg)
    getattr(train_loop, cfg.loop)(cfg)


if __name__ == '__main__':
    # main('dictionarywise_trainer.json', CFG)
    main('pairwise_trainer.json', CFG)
