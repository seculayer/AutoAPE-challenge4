import torch
from transformers import AutoTokenizer
# from model.metric import *


class CFG:
    """ Pipeline Setting """
    train, test = True, False
    checkpoint_dir = './saved/model'
    resume, state_dict = True, '/'
    name = 'FBP3_Base_Train_Pipeline'
    loop = 'mpl_loop'
    dataset = 'FBPDataset'  # dataset_class.dataclass.py -> FBPDataset, MPLDataset
    model_arch = 'FBPModel'  # model.model.py -> FBPModel, MPLModel
    model = 'microsoft/deberta-v3-large'
    tokenizer = AutoTokenizer.from_pretrained(model)
    pooling = 'MeanPooling'  # mean, attention, max, weightedlayer, concat, conv1d, lstm

    """ Common Options """
    wandb = True
    optuna = False  # if you want to tune hyperparameter, set True
    competition = 'FB3'
    seed = 42
    cfg_name = 'CFG'
    n_gpu = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_id = 0
    num_workers = 0

    """ Data Options """
    n_folds = 5
    max_len = 512
    epochs = 180
    batch_size = 64

    """ Gradient Options """
    amp_scaler = False
    gradient_checkpoint = True  # save parameter
    clipping_grad = True  # clip_grad_norm
    n_gradient_accumulation_steps = 1
    max_grad_norm = 1000

    """ Loss & Metrics Options """
    loss_fn = 'SmoothL1Loss'
    val_loss_fn = 'WeightedMSELoss'
    reduction = 'mean'
    metrics = ['MCRMSE', 'f_beta', 'recall']

    """ Optimizer with LLRD Options """
    optimizer = 'AdamW'  # options: SWA, AdamW
    llrd = True
    layerwise_lr = 5e-6
    layerwise_lr_decay = 0.9
    layerwise_weight_decay = 1e-2
    layerwise_adam_epsilon = 1e-6
    layerwise_use_bertadam = False
    betas = (0.9, 0.999)

    """ Scheduler Options """
    scheduler = 'cosine_annealing'  # options: cosine, linear, cosine_annealing, linear_annealing
    batch_scheduler = True
    num_cycles = 0.5  # num_warmup_steps = 0
    warmup_ratio = 0.1  # options: 0.05, 0.1

    """ SWA Options """
    swa = True
    swa_start = int(epochs*0.75)
    swa_lr = 1e-4
    anneal_epochs = 4
    anneal_strategy = 'cos'  # default = cos, available option: linear

    """ Model_Utils Options """
    stop_mode = 'min'
    freeze = False
    num_freeze = 2
    reinit = True
    num_reinit = 0
    awp = False
    nth_awp_start_epoch = 10
    awp_eps = 1e-2
    awp_lr = 1e-4