# %% [code]
# A solution for the Youtube8m-2019 Challenge.
# Explaination: https://www.kaggle.com/c/youtube8m-2019/discussion/112388

import multiprocessing
import os
import pickle
import time

from glob import glob
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from sklearn.model_selection import KFold
from scipy.stats import describe


SEGMENT_LEN     = 5
NUM_FOLDS       = 10
NUM_MODELS      = 10
BATCH_SIZE      = 96
LOG_FREQ        = 1000
NUM_CLASSES     = 1000
N_VIDEO_CLASSES = 3862
NUM_EPOCHS      = 25
FEATURES_DIM    = 1152

LEARNING_RATE   = 0.00011729283760398037
WEIGHT_DECAY    = 0.0011412688966608406

LR_FACTOR       = 0.1
LR_PATIENCE     = 3
LR_MINIMUM      = 3e-7
LR_THRESHOLD    = 1e-3


# Normally, training shows low GPU utilization (about 10%) since the training process is HDD-bound.
# However, only 10% of validation data is labeled.
# Let's compress it tight to optimize HDD access time (ofc training would be a lot faster with SSD).
def convert_val_data(prefix: str, wildcard: str) -> None:
    print('converting', wildcard)
    all_files = sorted(glob(wildcard))

    all_ids = []
    all_labels = []
    all_scores = []
    all_features_list = []

    for tfrec_file in all_files:
        for example in tf.python_io.tf_record_iterator(tfrec_file):
            tf_example = tf.train.Example.FromString(example)

            video_id = tf_example.features.feature['id'] \
                       .bytes_list.value[0].decode(encoding='utf-8')

            seg_start = list(tf_example.features.feature['segment_start_times'].int64_list.value)
            seg_labels = list(tf_example.features.feature['segment_labels'].int64_list.value)
            seg_scores = list(tf_example.features.feature['segment_scores'].float_list.value)

            tf_seq_example = tf.train.SequenceExample.FromString(example)
            num_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)

            if any(np.array(seg_start) > num_frames): # why are there videos with invalid labels?
                # print('skipping video', video_id, 'file', tfrec_file)
                continue

            for segment, label, score in zip(seg_start, seg_labels, seg_scores):
                features = []

                for frame in range(segment, segment + SEGMENT_LEN):
                    rgb = tf.decode_raw(tf_seq_example.feature_lists \
                                        .feature_list['rgb'].feature[frame] \
                                        .bytes_list.value[0],tf.uint8).numpy()
                    audio = tf.decode_raw(tf_seq_example.feature_lists \
                                          .feature_list['audio'].feature[frame] \
                                          .bytes_list.value[0],tf.uint8).numpy()

                    frame_features = np.concatenate([rgb, audio])
                    features.append(frame_features)

                all_ids.append(video_id)
                all_labels.append(label)
                all_scores.append(score)
                all_features_list.append(np.expand_dims(features, axis=0))


    all_features = np.concatenate(all_features_list)

    print('writing features to the disk')
    np.save(f'{prefix}_features.npy', all_features)

    print('writing labels to the disk')
    with open(f'{prefix}_ids.pkl', 'wb') as f:
        pickle.dump((all_ids, all_labels, all_scores), f)


def dequantize(feat_vector: np.array, max_quantized_value=2, min_quantized_value=-2) -> np.array:
    ''' Dequantize the feature from the byte format to the float format. '''
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


# PyTorch dataset class for numpy arrays.
class SegmentsDataset(torch.utils.data.Dataset):
    def __init__(self, ids: np.array, dataset_mask: Optional[np.array], labels: Optional[np.array],
                 scores: Optional[np.array], features_path: str, mode: str) -> None:
        print(f'creating SegmentsDataset in mode {mode}')

        self.ids = ids
        self.scores = scores
        self.mode = mode

        if self.mode != 'test':
            labels_table = pd.read_csv('/kaggle/input/youtube8m-2019/vocabulary.csv')
            labels_table = labels_table.Index

            encode_table = np.zeros(np.amax(labels_table) + 1, dtype=int)
            for i, index in enumerate(labels_table):
                encode_table[index] = i

            assert dataset_mask is not None and self.scores is not None
            self.features_indices = np.arange(dataset_mask.size)[dataset_mask]
            self.labels = encode_table[labels]
            features_size = dataset_mask.size

            assert self.labels.shape[0] == self.scores.shape[0]
            assert self.features_indices.size == self.labels.shape[0]
            assert features_size >= self.scores.shape[0]

            if self.mode == 'train':
                self.labels *= (self.scores > 0.5).astype(int)
        else:
            features_size = self.ids.shape[0]

        self.features = np.load(features_path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        features = self.features

        if self.mode != 'test':
            features_indices = self.features_indices
            labels = self.labels

            x = features[features_indices[index]]
        else:
            x = features[index]

        x = dequantize(x)
        x = torch.tensor(x, dtype=torch.float32)

        if self.mode == 'test':
            return x, 0
        else:
            y = labels[index].item()
            return x, y

    def __len__(self) -> int:
        return self.ids.shape[0]


# PyTorch dataset class for original TFRecords.
# I only use it for inference here, but it supports validation data with segment annotations.
def unwrap_video(video_dict):
    return (
        video_dict['id'],
        video_dict['labels'],
        video_dict['features'],
        video_dict['segment_start_times'],
        video_dict['segment_labels'],
        video_dict['segment_scores']
    )

def wrap_segment(vid, labels, start_time, features, segment_label, segment_score):
    return {
        'id': vid,
        'labels': labels,
        'start_time': start_time,
        'features': features,
        'segment_label': segment_label,
        'segment_score': segment_score
    }

class YouTube8MRecordParser:
    context_features = {
        "id": tf.io.FixedLenFeature((), tf.string),
        "labels": tf.io.VarLenFeature(tf.int64),
        "segment_start_times": tf.io.VarLenFeature(tf.int64),
        "segment_end_times": tf.io.VarLenFeature(tf.int64),
        "segment_labels": tf.io.VarLenFeature(tf.int64),
        "segment_scores": tf.io.VarLenFeature(tf.float32)
    }

    sequence_features = {
        "rgb": tf.io.FixedLenSequenceFeature([], tf.string),
        "audio": tf.io.FixedLenSequenceFeature([], tf.string),
    }

    @staticmethod
    def parse(proto):
        sample, sequence_parsed = tf.io.parse_single_sequence_example(
            proto,
            YouTube8MRecordParser.context_features,
            YouTube8MRecordParser.sequence_features
        )

        sample['features'] = tf.concat([
            tf.decode_raw(sequence_parsed['rgb'], tf.uint8),
            tf.decode_raw(sequence_parsed['audio'], tf.uint8)
        ], axis=-1
        )

        for k, v in sample.items():
            if k == 'labels' or 'segment' in k:
                sample[k] = v.values

        return sample

    @staticmethod
    def to_numpy(eager_sample):
        return {
            k: v.numpy()
            for k, v in eager_sample.items()
        }

    @staticmethod
    def get_video_dataset(tfrecords, num_workers=None):
        return tf.data.TFRecordDataset(tfrecords, num_parallel_reads=num_workers)\
            .map(YouTube8MRecordParser.parse, num_parallel_calls=num_workers)\
            .filter(lambda video: tf.math.greater_equal(tf.shape(video['features'])[0], 5))

    @staticmethod
    def _video_to_segments_iterator(vid, labels, features, segment_start_times, segment_labels, segment_scores):
        n_samples = len(features) // SEGMENT_LEN

        assert n_samples >= 5

        for idx in range(n_samples):
            start_time = SEGMENT_LEN * idx

            segment_label = segment_score = -1
            if start_time in segment_start_times:
                i = np.where(segment_start_times == start_time)[0][0]
                segment_label = segment_labels[i]
                segment_score = segment_scores[i]

            yield (
                vid,
                labels,
                start_time,
                features[start_time: start_time + SEGMENT_LEN],
                segment_label,
                np.float32(segment_score)
            )

    def _video_to_segments(*args):
        result = [[] for _ in range(6)]

        for segment in YouTube8MRecordParser._video_to_segments_iterator(*args):
            for i, value in enumerate(segment):
                result[i].append(value)

        return result

    @staticmethod
    def get_segment_dataset(tfrecords: List[str]) -> Any:
        return YouTube8MRecordParser.get_video_dataset(tfrecords, None)\
            .map(lambda video: tf.py_func(
                YouTube8MRecordParser._video_to_segments,
                unwrap_video(video),
                Tout=[tf.string, tf.int64, tf.int64, tf.uint8, tf.int64, tf.float32]),
                num_parallel_calls=None
            )\
            .flat_map(lambda *args: tf.data.Dataset.zip(tuple(
                tf.data.Dataset.from_tensor_slices(k)
                for k in args))
            )\
            .map(
                wrap_segment,
                num_parallel_calls=None
            )

class YouTube8MSegmentDataset(torch.utils.data.IterableDataset):
    def __init__(self, tfrecords: List[str]) -> None:
        self._dataset = YouTube8MRecordParser.get_segment_dataset(tfrecords)

    def __iter__(self) -> None:
        for i, segment in enumerate(map(YouTube8MRecordParser.to_numpy, self._dataset)):
            features = dequantize(segment['features']).astype(np.float32)
            yield features, segment['id'].decode()


def get_train_val_split(items: List[str], fold: int) -> Tuple[np.array, np.array]:
    skf = KFold(NUM_FOLDS, shuffle=True, random_state=0)
    items = np.array(items)
    train_idx, val_idx = list(skf.split(items))[fold]
    return items[train_idx], items[val_idx]

def load_train_data(fold: int) -> Any:
    with open('val_ids.pkl', 'rb') as f:
        all_ids, all_labels, all_scores = pickle.load(f)

    unique_ids = sorted(set(all_ids))
    unique_train_ids, unique_val_ids = get_train_val_split(unique_ids, fold)

    all_ids = np.array(all_ids)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    print(all_ids.shape)
    print(all_labels.shape)

    train_mask = np.isin(all_ids, unique_train_ids)
    train_ids = all_ids[train_mask]
    train_labels = all_labels[train_mask]
    train_scores = all_scores[train_mask]

    val_ids = all_ids[~train_mask]
    val_labels = all_labels[~train_mask]
    val_scores = all_scores[~train_mask]

    train_dataset = SegmentsDataset(train_ids, train_mask, train_labels, train_scores,
                                    'val_features.npy', mode='train')

    val_dataset = SegmentsDataset(val_ids, ~train_mask, val_labels, val_scores,
                                  'val_features.npy', mode='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, drop_last=False)

    return train_loader, val_loader

def load_test_data(wildcard: str) -> Any:
    test_dataset = YouTube8MSegmentDataset(glob(wildcard))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return test_loader


class AverageMeter:
    ''' Computes and stores the average and current value. '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_lr(optimizer: Any, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer: Any) -> float:
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        return lr

    assert False

def accuracy(predicts: Any, targets: Any) -> float:
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    if len(predicts.shape) == 2:
        predicts = np.argmax(predicts, axis=1)

    if len(targets.shape) == 2:
        targets = np.argmax(targets, axis=1)

    if predicts.shape != targets.shape:
        print(predicts.shape)
        print(targets.shape)
        assert False

    return np.mean(predicts == targets)

def average_precision(actuals, predictions, k=None):
    num_positives = actuals.sum() + 1e-10

    sorted_idx = np.argsort(predictions)[::-1]
    if k is not None:
        sorted_idx = sorted_idx[:k]

    actuals = actuals[sorted_idx]
    precisions = np.cumsum(actuals) / np.arange(1, len(actuals) + 1)

    return (precisions * actuals).sum() / float(num_positives)

class MeanAveragePrecisionCalculator:
    ''' Classwise MAP@K - metric for Youtube-8M 2019 competition. '''

    def __init__(self, num_classes=NUM_CLASSES, k=10 ** 5):
        self._num_classes = num_classes
        self._k = k
        self._predictions = [[] for _ in range(num_classes)]
        self._actuals = [[] for _ in range(num_classes)]

    def accumulate(self, predictions, actuals, masks=None):
        if masks is None:
            masks = np.ones_like(actuals)

        for i in range(self._num_classes):
            mask = masks[:, i] > 0

            self._predictions[i].append(predictions[:, i][mask])
            self._actuals[i].append(actuals[:, i][mask])

    def __call__(self):
        aps = []
        positive_count = []
        total_count = []

        for i in range(self._num_classes):
            actuals = np.concatenate(self._actuals[i])
            predictions = np.concatenate(self._predictions[i])

            aps.append(average_precision(actuals, predictions, self._k))

            total_count.append(len(actuals))
            positive_count.append(actuals.sum())

        return np.mean(aps)


def train_epoch(train_loader: Any, model: Any, criterion: Any, optimizer: Any,
                epoch: int, lr_scheduler: Any) -> float:
    print(f'epoch: {epoch}')
    print(f'learning rate: {get_lr(optimizer)}')

    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)

    print(f'total batches: {num_steps}')
    end = time.time()
    activation = nn.Softmax(dim=1)

    for i, (input_, target) in enumerate(train_loader):
        input_ = input_.cuda()
        output = model(input_)

        loss = criterion(output, target.cuda())

        predict = torch.argmax(output.detach(), dim=-1)
        avg_score.update(accuracy(predict, target))

        losses.update(loss.data.item(), input_.size(0))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % LOG_FREQ == 0:
            print(f'{epoch} [{i}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'acc {avg_score.val:.4f} ({avg_score.avg:.4f})')

    print(f' * average acc on train {avg_score.avg:.4f}')
    return avg_score.avg

def inference(data_loader: Any, model: Any) -> np.array:
    ''' Returns predictions array. '''
    model.eval()

    predicts_list = []
    activation = nn.Softmax(dim=1)

    with torch.no_grad():
        for input_, target in data_loader:
            output = model(input_.cuda())
            output = activation(output)
            predicts_list.append(output.detach().cpu().numpy())

    predicts = np.concatenate(predicts_list)
    print('predicts', predicts.shape)
    return predicts

def validate(val_loader: Any, model: Any, epoch: int) -> float:
    ''' Infers predictions and calculates validation score. '''
    print('validate()')
    val_pred = inference(val_loader, model)

    metric = MeanAveragePrecisionCalculator()

    val_true = val_loader.dataset.labels
    val_scores = val_loader.dataset.scores

    assert val_true.size == val_pred.shape[0]

    masks = np.eye(NUM_CLASSES)[val_true]   # convert to one-hot encoding
    actuals = masks * np.expand_dims(val_scores, axis=-1)

    metric.accumulate(val_pred, actuals, masks)
    score = metric()

    print(f' * epoch {epoch} validation score: {score:.4f}')
    return score

# In my pipeline, there is a single inference function for both validation and test set prediction.
# But I had to copy-paste this function here to add some hacks to bypass
# Kaggle kernel memory restrictions.
def inference_for_testset(test_predicts: np.array, data_loader: Any, model: Any) -> np.array:
    ''' Returns predictions array. '''
    model.eval()

    ids_list: List[str] = []
    activation = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (input_, ids) in enumerate(data_loader):
            output = model(input_.cuda())
            output = activation(output)

            ids_list.extend(ids)
            pred = output.detach().cpu().numpy()
            bs = data_loader.batch_size
            test_predicts[i * bs : i * bs + pred.shape[0]] += pred

    ids = np.array(ids_list)
    print('ids', ids.shape)
    return ids


class SwishActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(x) * x

class ClassifierModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        width = FEATURES_DIM

        for num_neurons in [2765, 1662]:
            layers.append(nn.Linear(width, num_neurons))
            width = num_neurons

            layers.append(nn.BatchNorm1d(width))
            layers.append(SwishActivation())

        layers.append(nn.Linear(width, NUM_CLASSES))
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.layers(x)
        return x


def get_model_path(fold_num: int) -> str:
    return f'best_model_fold_{fold_num}.pth'

def train_model(fold_num: int) -> float:
    print('=' * 80)
    print(f'training a model, fold {fold_num}')

    model = ClassifierModel()
    model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loader, val_loader = load_train_data(fold_num)
    lr_scheduler = lr_sched.ReduceLROnPlateau(optimizer, mode='max', factor=LR_FACTOR,
                                      patience=LR_PATIENCE, threshold=LR_THRESHOLD,
                                      min_lr=LR_MINIMUM)

    last_epoch = -1
    print(f'training will start from epoch {last_epoch + 1}')

    best_score = 0.0
    best_epoch = 0

    last_lr = get_lr(optimizer)
    best_model_path = None

    for epoch in range(last_epoch + 1, NUM_EPOCHS):
        print('-' * 50)
        lr = get_lr(optimizer)

        # if we have just reduced LR, reload the best saved model
        if lr < last_lr - 1e-10 and best_model_path is not None:
            print(f'learning rate dropped: {lr}, reloading')
            last_checkpoint = torch.load(best_model_path)

            model.load_state_dict(last_checkpoint['state_dict'])
            optimizer.load_state_dict(last_checkpoint['optimizer'])
            print(f'checkpoint loaded: {best_model_path}')
            set_lr(optimizer, lr)
            last_lr = lr

        train_epoch(train_loader, model, criterion, optimizer, epoch, lr_scheduler)
        score = validate(val_loader, model, epoch)

        lr_scheduler.step(metrics=score)

        is_best = score > best_score
        best_score = max(score, best_score)
        if is_best:
            best_epoch = epoch

        if is_best:
            best_model_path = get_model_path(fold_num)

            data_to_save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            torch.save(data_to_save, best_model_path)
            print(f'a snapshot was saved to {best_model_path}')

    print(f'best score: {best_score:.04f}')
    return -best_score

def predict_with_model(test_predicts: np.array, fold_num: int) -> np.array:
    print(f'predicting on the test set, fold {fold_num}')

    model = ClassifierModel()
    model.cuda()

    best_model_path = get_model_path(fold_num)

    if os.path.exists(best_model_path):
        print(f'loading checkpoint: {best_model_path}')
        last_checkpoint = torch.load(best_model_path)
        model.load_state_dict(last_checkpoint['state_dict'])

        last_epoch = last_checkpoint['epoch']
        print(f'loaded the model from epoch {last_epoch}')
        os.unlink(best_model_path)

    output = inference_for_testset(test_predicts, test_loader, model)
    return output

def generate_submission(ids: np.array, probas: np.array) -> None:
    last_id, current_seg = None, 0
    segment_start_times = []
    print('generating submission')

    for video_id in ids:
        if video_id == last_id:
            current_seg += 1
        else:
            current_seg = 0
            last_id = video_id

        segment_start_times.append(current_seg * 5)

    labels = pd.read_csv('/kaggle/input/youtube8m-2019/vocabulary.csv')
    classes_table = labels.Index

    assert probas.shape[1] == NUM_CLASSES
    max_predicts = 10 ** 5

    with open('submission.csv', 'w') as fout:
        fout.write('Class,Segments\n')

        for i in range(NUM_CLASSES):
            current_probas = probas[:, i]
            sorted_idx = np.argsort(current_probas)[::-1][:max_predicts]
            current_line = [
                '{}:{}'.format(ids[idx], segment_start_times[idx])
                for idx in sorted_idx
            ]

            fout.write('{},{}\n'.format(classes_table[i], ' '.join(current_line)))

    print('submission has been generated')

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()

    convert_val_data('val', '/kaggle/input/youtube-challenge-2019-val/val/validate*.tfrecord')
    test_loader = load_test_data('/kaggle/input/youtubechallenge2019test/test/test*.tfrecord')
    test_predicts = np.zeros((2038114, 1000), dtype=np.float16)

    for fold_idx in range(NUM_MODELS):
        train_model(fold_idx)
        test_ids = predict_with_model(test_predicts, fold_idx)

    test_predicts /= NUM_MODELS
    generate_submission(test_ids, test_predicts)