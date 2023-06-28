import gc
import wandb
from configuration import CFG
from trainer import *
from utils.helper import class2dict

g = torch.Generator()
g.manual_seed(CFG.seed)


def train_loop(cfg: any) -> None:
    """ Base Trainer Loop Function """
    tmp_valid = pd.read_csv('./dataset_class/data_folder/train.csv')
    fold_list = [i for i in range(cfg.n_folds)]
    for fold in tqdm(fold_list):
        print(f'============== {fold}th Fold Train & Validation ==============')
        wandb.init(
            project=cfg.name,
            name=f'FBP2_fold{fold}/' + cfg.model,
            config=class2dict(cfg),
            group=f'FBP2/{cfg.model}',
            job_type='train',
            entity="qcqced"
        )
        early_stopping = EarlyStopping(mode=cfg.stop_mode, patience=3)
        early_stopping.detecting_anomaly()

        val_score_max = get_save_thresholds(cfg)
        train_input = getattr(trainer, cfg.name)(cfg, g)  # init object
        loader_train, loader_valid, train, valid = train_input.make_batch(fold)
        model, criterion, val_metrics, optimizer, lr_scheduler = train_input.model_setting(len(train))

        for epoch in range(cfg.epochs):
            print(f'[{epoch + 1}/{cfg.epochs}] Train & Validation')
            train_loss, train_accuracy, train_recall, train_precision = train_input.train_fn(
                loader_train, model, criterion, optimizer, lr_scheduler, val_metrics
            )
            val_ids_list, val_pred_list = train_input.valid_fn(
                loader_valid, model
            )
            # 1) make prediction dataframe
            final_pred = []
            for i in range(len(valid)):
                idx = valid.id.values[i]
                pred = val_pred_list[i]
                tmp_pred = []
                j = 0
                while j < len(pred):
                    cls = pred[j]
                    if cls == 'O':
                        j += 1
                    else:
                        cls = cls.replace('B', 'I')  # spans start with B
                    end = j + 1
                    while end < len(pred) and pred[end] == cls:
                        end += 1

                    if cls != 'O' and cls != '' and end - j > 7:
                        final_pred.append(
                            (idx, cls.replace('I-', ''),
                             ' '.join(map(str, list(range(j, end)))))
                        )
                    j = end
            pred_df = pd.DataFrame(final_pred)
            pred_df.columns = ['id', 'class', 'predictionstring']

            # 2) calculate cross validation score
            batch_valid = tmp_valid.loc[tmp_valid['id'].isin(val_ids_list)].copy()
            f1_list = []
            unique_class = pred_df['class'].unique()
            for i, c in enumerate(unique_class):
                print(f'iteration: {i}, class: {c}')
                subset_pred_df = pred_df.loc[pred_df['class'] == c].copy()
                gt_df = batch_valid.loc[batch_valid['discourse_type'] == c].copy()
                f1_score = calculate_f1(subset_pred_df, gt_df)
                print(c, f1_score)  # print f1 score for each class
                f1_list.append(f1_score)
            final_f1_score = np.mean(f1_list)  # average == 'micro'
            wandb.log({
                '<epoch> Train Loss': train_loss,
                '<epoch> Train Accuracy': train_accuracy,
                '<epoch> Train Recall': train_recall,
                '<epoch> Train Precision': train_precision,
                '<epoch> CV Score (Comp F1)': final_f1_score,
            })
            print(f'[{epoch + 1}/{cfg.epochs}] Train Loss: {np.round(train_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Train Accuracy: {np.round(train_accuracy, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Train Recall: {np.round(train_recall, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Train Precision: {np.round(train_precision, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] CV Score (Comp F1): {np.round(final_f1_score, 4)}')

            if val_score_max <= final_f1_score:
                print(f'[Update] Valid Score : ({val_score_max:.4f} => {final_f1_score:.4f}) Save Parameter')
                print(f'Best Score: {final_f1_score}')
                torch.save(model.state_dict(),
                           f'{cfg.checkpoint_dir}fold{fold}_{get_name(cfg)}_state_dict.pth')
                val_score_max = final_f1_score

            # Check if Trainer need to Early Stop
            early_stopping(final_f1_score)
            if early_stopping.early_stop:
                break
            del train_loss, train_accuracy, train_recall, train_precision, final_f1_score, subset_pred_df, batch_valid
            gc.collect(), torch.cuda.empty_cache()

        del model, loader_train, loader_valid, train, valid  # delete for next fold
        gc.collect(), torch.cuda.empty_cache()
        wandb.finish()
