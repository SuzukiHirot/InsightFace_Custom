import numbers
import os
import queue as Queue
import threading
from typing import Iterable

#import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn

from torchvision.datasets.folder import default_loader #png のため実装


def get_dataloader(
    cfg,
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    dali_aug = False,
    seed = 2048,
    num_workers = 2,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Synthetic
    if root_dir == "synthetic":
        import mxnet as mx  # to avoid unused import warning
        train_set = SyntheticDataset()
        dali = False

    # Mxnet RecordIO
    elif os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)

    # vface10k (.txt ファイル) の場合
    elif root_dir.endswith(".txt"):
        if not hasattr(cfg, 'data_root') or cfg.data_root is None:
            raise ValueError("config.rec is a .txt file, but config.data_root (image folder) is not set in config.")
        

        # 1. ランダムクロップ (論文指定) + 低解像度変換 (Spec ①: 50%-80% scale)
        #    torchvision の RandomResizedCrop でこれらを同時に実現
        #    (InsightFace の標準入力サイズは 112x112 です)
        transform_list = [
            transforms.RandomResizedCrop(
                size=(112, 112),  # InsightFace の標準入力サイズ
                scale=(0.7, 1.0), # Spec ①: 50%-80% スケール
                ratio=(0.75, 1.33) # 標準的なアスペクト比
            ),
            # 2. 水平方向ランダムフリップ (50% prob)
            transforms.RandomHorizontalFlip(p=0.5),
            # 3. フォトメトリックオーグメンテーション (Spec ③: 実行50%)
            #    明るさ(brightness) [0.8-1.2],コントラスト, 彩度(saturation) [0.8-1.2], 色相(hue) [-0.05, 0.05]
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04)
            ], p=0.7), # Spec ③: 実行50%
            # 4. 低解像度変換 の「ガウシアンブラー」部分 (Spec ①: 実行50%)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3) # 3x3 カーネル
            ], p=0.5), # Spec ①: 実行50%

            # 5. PIL Image を Tensor に変換 (0-255 -> 0.0-1.0)
            #    RandomErasing の前に実行する必要がある
            transforms.ToTensor(),
            # 6. ランダムイレース (Spec ②: 実行50%)
            #    scale=(0.02, 0.40) は 2%-40% の領域を消去
            transforms.RandomErasing(
                p=0.7,           # Spec ②: 実行50%
                scale=(0.02, 0.30), # Spec ②: 2%-40%
                ratio=(0.3, 3.3),   # FaceX-Zooコードの r1, r2 に相当
                value=0, 
                inplace=False
            ),
            # 7. 正規化 (-1.0 ~ 1.0 の範囲に)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        transform = transforms.Compose(transform_list)
        # === データ拡張の定義ここまで ===

        
        print(f"Using TxtDataset for list file: {root_dir}")
        train_set = TxtDataset(root_txt_file=root_dir, data_root=cfg.data_root, transform=transform)

    # Image Folder (従来の動作)
    else:

        # 1. ランダムクロップ (論文指定) + 低解像度変換 (Spec ①: 50%-80% scale)
        #    torchvision の RandomResizedCrop でこれらを同時に実現
        #    (InsightFace の標準入力サイズは 112x112 です)
        transform_list = [
            transforms.RandomResizedCrop(
                size=(112, 112),  # InsightFace の標準入力サイズ
                scale=(0.7, 1.0), # Spec ①: 50%-80% スケール
                ratio=(0.75, 1.33) # 標準的なアスペクト比
            ),

            # 2. 水平方向ランダムフリップ (50% prob)
            transforms.RandomHorizontalFlip(p=0.5),

            # 3. フォトメトリックオーグメンテーション (Spec ③: 実行50%)
            #    明るさ(brightness) [0.8-1.2], 彩度(saturation) [0.8-1.2]
            #    FaceX-Zooコードを参考にコントラストも追加
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04)
            ], p=0.7), # Spec ③: 実行50%

            # 4. 低解像度変換 の「ガウシアンブラー」部分 (Spec ①: 実行50%)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3) # 3x3 カーネル
            ], p=0.5), # Spec ①: 実行50%

            # 5. PIL Image を Tensor に変換 (0-255 -> 0.0-1.0)
            #    RandomErasing の前に実行する必要がある
            transforms.ToTensor(),
            
            # 6. ランダムイレース (Spec ②: 実行50%)
            #    scale=(0.02, 0.40) は 2%-40% の領域を消去
            transforms.RandomErasing(
                p=0.7,           # Spec ②: 実行50%
                scale=(0.02, 0.30), # Spec ②: 2%-40%
                ratio=(0.3, 3.3),   # FaceX-Zooコードの r1, r2 に相当
                value=0, 
                inplace=False
            ),

            # 7. 正規化 (-1.0 ~ 1.0 の範囲に)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        transform = transforms.Compose(transform_list)
        # === データ拡張の定義ここまで ===

        print(f"Using default ImageFolder for directory: {root_dir}")
        train_set = ImageFolder(root_dir, transform)
    # DALI
    if dali:
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank, dali_aug=dali_aug)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader

# === ここから追記 (vface10k 用のカスタムデータセット) ===# dataset.py 内の TxtDataset クラスを修正

class TxtDataset(Dataset):
    def __init__(self, root_txt_file, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.loader = default_loader
        self.samples = []

        with open(root_txt_file, 'r') as f:
            for line in f:
                path, label = line.strip().split(' ')
                # パスを保持しておく
                self.samples.append((os.path.join(self.data_root, path), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform:
                sample = self.transform(sample)
            
            # 【重要】戻り値を3つにする (画像, ラベル, パス)
            return sample, label, path 
            
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            path, label = self.samples[0]
            sample = self.loader(path)
            if self.transform:
                sample = self.transform(sample)
            
            # エラー時も3つ返す
            return sample, label, path
# === 追記ここまで ===

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                if torch.is_tensor(self.batch[k]):
                    self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5),
    dali_aug=False
    ):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    def dali_random_resize(img, resize_size, image_size=112):
        img = fn.resize(img, resize_x=resize_size, resize_y=resize_size)
        img = fn.resize(img, size=(image_size, image_size))
        return img
    def dali_random_gaussian_blur(img, window_size):
        img = fn.gaussian_blur(img, window_size=window_size * 2 + 1)
        return img
    def dali_random_gray(img, prob_gray):
        saturate = fn.random.coin_flip(probability=1 - prob_gray)
        saturate = fn.cast(saturate, dtype=types.FLOAT)
        img = fn.hsv(img, saturation=saturate)
        return img
    def dali_random_hsv(img, hue, saturation):
        img = fn.hsv(img, hue=hue, saturation=saturation)
        return img
    def multiplexing(condition, true_case, false_case):
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    condition_resize = fn.random.coin_flip(probability=0.1)
    size_resize = fn.random.uniform(range=(int(112 * 0.5), int(112 * 0.8)), dtype=types.FLOAT)
    condition_blur = fn.random.coin_flip(probability=0.2)
    window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)
    condition_flip = fn.random.coin_flip(probability=0.5)
    condition_hsv = fn.random.coin_flip(probability=0.2)
    hsv_hue = fn.random.uniform(range=(0., 20.), dtype=types.FLOAT)
    hsv_saturation = fn.random.uniform(range=(1., 1.2), dtype=types.FLOAT)

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        if dali_aug:
            images = fn.cast(images, dtype=types.UINT8)
            images = multiplexing(condition_resize, dali_random_resize(images, size_resize, image_size=112), images)
            images = multiplexing(condition_blur, dali_random_gaussian_blur(images, window_size_blur), images)
            images = multiplexing(condition_hsv, dali_random_hsv(images, hsv_hue, hsv_saturation), images)
            images = dali_random_gray(images, 0.1)

        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()
