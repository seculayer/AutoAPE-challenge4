import torch
from model.model import NetworkB6, NetworkCait
import ttach as tta
import pandas as pd
import numpy as np
from utils.seed import fix_seed
from utils.dataset import CustomDataset
from utils.path import load_train_test ,label_unique
from torch.utils.data import DataLoader
import gc

fix_seed(2023)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 4
train_imgs , test_imgs = load_train_test()
# Test
test_dataset = CustomDataset(np.array(test_imgs), np.array(["tmp"]*len(test_imgs)), mode='test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

tta_transforms = tta.Compose(
    [
        tta.Rotate90(angles=[0, 90, 180, 270]),
        tta.Multiply([0.9, 1])
    ]
)
gc.collect()
torch.cuda.empty_cache()

loaded_model = torch.load('weights/Cait_batch8_lr_model/Cait_399_model.pt')
model_b3 = NetworkCait().to(device)
model_b3.load_state_dict(loaded_model['model_state_dict'])

tta_model_b3 = tta.ClassificationTTAWrapper(model_b3, tta_transforms)

tta_model_b3.eval()
f_pred = []

with torch.no_grad():
    for batch in (test_loader):
        x = torch.tensor(batch[0]['image'], dtype = torch.float32, device = device)
        with torch.cuda.amp.autocast():
            # ensemble
            pred_b3 = tta_model_b3(x)
            pred = pred_b3
        f_pred.extend(pred.argmax(1).detach().cpu().numpy().tolist())

label_decoder = {val:key for key, val in label_unique.items()}
f_result = [label_decoder[result] for result in f_pred]

submission = pd.read_csv("data/open/sample_submission.csv")
submission["label"] = f_result

submission.to_csv('submission/Cait_batch4_model_submission.csv',index=False)
