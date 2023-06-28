import os, warnings
import trainer.train_loop as train_loop
from omegaconf import OmegaConf
from configuration import CFG
from utils.helper import check_library, all_type_seed
from utils import sync_config
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ['LRU_CACHE_CAPACITY'] = "1"
print(check_library(False))
all_type_seed(CFG, True)


def main(config_path: str, cfg) -> None:
    sync_config(OmegaConf.load(config_path))
    getattr(train_loop, cfg.loop)(cfg)


if __name__ == '__main__':
    main('fbp2_DeBERTa_V3_config.json', CFG)
    # main('fbp2_LongFormer_config.json', CFG)
