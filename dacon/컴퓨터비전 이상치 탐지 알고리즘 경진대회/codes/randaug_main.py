import numpy as np
import torch
import torch.nn as nn
from utils.seed import fix_seed
from utils.dataset import CustomRandaugDataset
from utils.path import load_train_test ,train_labels
from torch.utils.data import DataLoader
from model.model import NetworkB3
from sklearn.metrics import f1_score, accuracy_score
import ttach as tta

import time
import os
import gc

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

fix_seed(2023)

batch_size = 16
epochs = 100
train_imgs, test_imgs = load_train_test()

uni_list = [1,2]
mag_list = [2,6,10,14]

for uni in uni_list:
    for mag in mag_list:

        # Train
        train_dataset = CustomRandaugDataset(np.array(train_imgs),
                                             np.array(train_labels),
                                             mode='train',
                                             N=uni,
                                             M=mag)

        train_loader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=batch_size)

        # Test
        test_dataset = CustomRandaugDataset(np.array(test_imgs), np.array(["tmp"]*len(test_imgs)), mode='test',N=uni,M=mag)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

        def score_function(real, pred):
            score = f1_score(real, pred, average="macro")
            return score

        model_b3 = NetworkB3().to(device)

        optimizer = torch.optim.Adam(model_b3.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        path = f'weights/n_{uni}_m_{mag}_model'

        gc.collect()
        torch.cuda.empty_cache()

        if not os.path.isdir(path):
            os.mkdir(path)

        for epoch in range(epochs):
            start=time.time()
            train_loss = 0
            train_pred=[]
            train_y=[]
            model_b3.train()
            for batch in (train_loader):
                optimizer.zero_grad()
                x = torch.tensor(batch[0]['image'], dtype=torch.float32, device=device)
                y = torch.tensor(batch[1], dtype=torch.long, device=device)
                with torch.cuda.amp.autocast():
                    pred = model_b3(x)
                loss = criterion(pred, y)


                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()/len(train_loader)
                train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                train_y += y.detach().cpu().numpy().tolist()

            train_f1 = score_function(train_y, train_pred)

            TIME = time.time() - start
            print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
            print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')

            # 모델 저장
            # torch.save({
            #             'epoch': epoch,
            #             'model_state_dict': model_b3.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             "scaler": scaler.state_dict(),
            #             'loss': loss,
            #             }, f"{path}/b3_model.pt")
            if ((epoch+1)%10)==0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_b3.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        'loss': loss,
                    }, f"{path}/b3_{epoch}_model.pt")
        try:

            tta_transforms = tta.Compose(
                [
                    tta.Rotate90(angles=[0, 90, 180, 270]),
                    tta.Multiply([0.9, 1])
                ]
            )

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

            submission.to_csv(f"{path}/n_{uni}_m_{mag}_model_submission.csv",index=False)
        except:
            pass

