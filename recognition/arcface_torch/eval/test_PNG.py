# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import numpy as np
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from PIL import Image
import csv
import datetime
import sys
from scipy import interpolate
import scipy.io as scio
from abc import ABCMeta, abstractmethod
import cv2 

# Add parent directory to path to import backbones
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backbones.iresnet import get_model
from backbones.iresnet import SEModule

# MXNet import for legacy support (optional)
try:
    import mxnet as mx
    from mxnet import ndarray as nd
except ImportError:
    pass

# ========= (Evaluation Helper Functions) =========
class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            epsilon = 1e-8
            far_train = far_train + np.arange(len(far_train)) * epsilon
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
                                       np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
                                      np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


# ========= (PairsParser Classes - Modified for PNG) =========

class PairsParser(metaclass=ABCMeta):
    """Parse the pair list for lfw based protocol."""
    def __init__(self, pairs_file):
        self.pairs_file = pairs_file
    
    @abstractmethod
    def parse_pairs(self):
        """The method for parsing pair list."""
        pass

class LFW_PairsParser(PairsParser):
    """The pairs parser for lfw."""
    def parse_pairs(self):
        test_pair_list = []
        with open(self.pairs_file, 'r') as pairs_file_buf:
            line = pairs_file_buf.readline() # skip first line (header)
            
            # 簡易ヘッダーチェック: ヘッダーじゃなければ巻き戻す等の処理も可能だが、
            # 標準LFW形式は1行目ヘッダーなのでスキップ前提で進める。
            
            line = pairs_file_buf.readline().strip()
            while line:
                line_strs = line.split('\t')
                
                # タブ区切りでない場合はスペース区切りを試す
                if len(line_strs) < 3:
                    line_strs = line.split(' ')

                if len(line_strs) == 3:
                    person_name = line_strs[0]
                    image_index1 = line_strs[1]
                    image_index2 = line_strs[2]
                    # 【変更】PNG評価用スクリプトなので .png に変更
                    image_name1 = person_name + '/' + person_name + '_' + image_index1.zfill(4) + '.png'
                    image_name2 = person_name + '/' + person_name + '_' + image_index2.zfill(4) + '.png'
                    label = 1
                elif len(line_strs) == 4:
                    person_name1 = line_strs[0]
                    image_index1 = line_strs[1]
                    person_name2 = line_strs[2]
                    image_index2 = line_strs[3]
                    # 【変更】PNG評価用スクリプトなので .png に変更
                    image_name1 = person_name1 + '/' + person_name1 + '_' + image_index1.zfill(4) + '.png'
                    image_name2 = person_name2 + '/' + person_name2 + '_' + image_index2.zfill(4) + '.png'
                    label = 0
                else:
                    # 空行などはスキップ
                    if not line: break
                    print(f"Skipping line: {line}")
                    line = pairs_file_buf.readline().strip()
                    continue
                
                test_pair_list.append((image_name1, image_name2, label))
                line = pairs_file_buf.readline().strip()
        return test_pair_list

class RFW_PairsParser(PairsParser):
    def parse_pairs(self):
        test_pair_list = []
        with open(self.pairs_file, 'r') as pairs_file_buf:
            line = pairs_file_buf.readline().strip()
            while line:
                line_strs = line.split('\t')
                if len(line_strs) == 3:
                    person_name = line_strs[0]
                    image_index1 = line_strs[1]
                    image_index2 = line_strs[2]
                    image_name1 = person_name + '/' + person_name + '_' + image_index1.zfill(4) + '.png'
                    image_name2 = person_name + '/' + person_name + '_' + image_index2.zfill(4) + '.png'
                    label = 1
                elif len(line_strs) == 4:
                    person_name1 = line_strs[0]
                    image_index1 = line_strs[1]
                    person_name2 = line_strs[2]
                    image_index2 = line_strs[3]
                    image_name1 = person_name1 + '/' + person_name1 + '_' + image_index1.zfill(4) + '.png'
                    image_name2 = person_name2 + '/' + person_name2 + '_' + image_index2.zfill(4) + '.png'
                    label = 0
                else:
                    raise Exception('Line error: %s.' % line)
                test_pair_list.append((image_name1, image_name2, label))
                line = pairs_file_buf.readline().strip()
        return test_pair_list

class CPLFW_PairsParser(PairsParser):
    """The pairs parser for cplfw."""
    def parse_pairs(self):        
        pair_list = []
        with open(self.pairs_file, 'r') as pairs_file_buf:
            line1 = pairs_file_buf.readline().strip()
            while line1:
                line2 = pairs_file_buf.readline().strip()
                if not line2: break
                image_name1 = line1.split(' ')[0]
                image_name2 = line2.split(' ')[0]
                label = line1.split(' ')[1]
                pair_list.append((image_name1, image_name2, int(label)))
                line1 = pairs_file_buf.readline().strip()
        
        # 6000ペア未満の場合のクラッシュ防止
        if len(pair_list) == 6000:
            test_pair_list = []
            positive_start = 0 
            negtive_start = 3000 
            for set_idx in range(10):
                positive_index = positive_start + 300 * set_idx
                negtive_index = negtive_start + 300 * set_idx
                cur_positive_pair_list = pair_list[positive_index : positive_index + 300]
                cur_negtive_pair_list = pair_list[negtive_index : negtive_index + 300]
                test_pair_list.extend(cur_positive_pair_list)
                test_pair_list.extend(cur_negtive_pair_list)
            return test_pair_list
        else:
            return pair_list

class CALFW_PairsParser(PairsParser):
    """The pairs parser for calfw."""
    def parse_pairs(self):
        pair_list = []
        with open(self.pairs_file, 'r') as pairs_file_buf:
            line1 = pairs_file_buf.readline().strip()
            while line1:
                line2 = pairs_file_buf.readline().strip()
                if not line2: break
                image_name1 = line1.split(' ')[0]
                image_name2 = line2.split(' ')[0]
                label = int(line1.split(' ')[1])
                if label != 0:
                    label = 1
                pair_list.append((image_name1, image_name2, label))
                line1 = pairs_file_buf.readline().strip()
        
        if len(pair_list) == 6000:
            test_pair_list = []
            positive_start = 0 
            negtive_start = 3000 
            for set_idx in range(10):
                positive_index = positive_start + 300 * set_idx
                negtive_index = negtive_start + 300 * set_idx
                cur_positive_pair_list = pair_list[positive_index : positive_index + 300]
                cur_negtive_pair_list = pair_list[negtive_index : negtive_index + 300]
                test_pair_list.extend(cur_positive_pair_list)
                test_pair_list.extend(cur_negtive_pair_list)
            return test_pair_list
        else:
            return pair_list

class AgeDB_PairsParser(PairsParser):
    """The pairs parser for agedb."""
    def parse_pairs(self):
        test_pair_list = []
        pairs_data = scio.loadmat(self.pairs_file)
        splits = pairs_data['splits']
        for split_index in range(10):
            cur_split = splits[split_index]
            cur_pairs = cur_split[0][0][0][0]
            cur_labels = cur_split[0][0][0][1][0]
            cur_first_list = cur_pairs[0]
            cur_second_list = cur_pairs[1]
            for pair_index in range(600):
                cur_first = cur_first_list[pair_index]
                # 【変更】PNG評価用スクリプトなので .png に変更
                cur_first_name = cur_first[0][0][0][0] + '.png'
                cur_second = cur_second_list[pair_index]
                cur_second_name = cur_second[0][0][0][0] + '.png'
                cur_label = cur_labels[pair_index]
                if cur_label == -1:
                    cur_label = 0
                test_pair_list.append((cur_first_name, cur_second_name, cur_label))
        return test_pair_list

class iCarB_PairsParser(PairsParser):
    def parse_pairs(self):
        test_pair_list = []
        with open(self.pairs_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    pid, b1, b2 = parts
                    img1 = f"{pid}/{b1}.png"
                    img2 = f"{pid}/{b2}.png"
                    label = 1
                elif len(parts) == 4:
                    pid1, b1, pid2, b2 = parts
                    img1 = f"{pid1}/{b1}.png"
                    img2 = f"{pid2}/{b2}.png"
                    label = 0
                else:
                    continue 
                test_pair_list.append((img1, img2, label))
        return test_pair_list

class PairsParserFactory(object):
    def __init__(self, pairs_file, test_set):
        self.pairs_file = pairs_file
        self.test_set = test_set
    def get_parser(self):
        # 大文字小文字を吸収して判定
        t = self.test_set.upper()
        if t == 'LFW':
            return LFW_PairsParser(self.pairs_file)
        elif t == 'CPLFW':
            return CPLFW_PairsParser(self.pairs_file)
        elif t == 'CALFW':
            return CALFW_PairsParser(self.pairs_file)
        elif t == 'AGEDB30' or t == 'AGEDB':
            return AgeDB_PairsParser(self.pairs_file)
        elif 'RFW' in t:
            return RFW_PairsParser(self.pairs_file)
        elif t == 'ICARB-FACE':
            return iCarB_PairsParser(self.pairs_file)
        else:
            return None

# ========= (Loading Logic Integration) =========

def read_pairs_from_csv(pairs_csv):
    pairs = []
    with open(pairs_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img1 = row.get('img1') or row.get('image1') or row.get('path1')
            img2 = row.get('img2') or row.get('image2') or row.get('path2')
            issame = row.get('issame') or row.get('same')
            if img1 is None or img2 is None:
                raise ValueError("CSV must contain img1 and img2 columns")
            issame_bool = str(issame).lower() in ('1','true','yes','t')
            pairs.append((img1, img2, issame_bool))
    return pairs
@torch.no_grad()
def load_images_from_pairs(images_dir, pairs_path, image_size, pairs_format='LFW'):
    img_h, img_w = image_size[0], image_size[1]
    pairs = []
    
    # 1. Parsing Pairs
    if pairs_path.lower().endswith('.csv'):
        raw_pairs = read_pairs_from_csv(pairs_path)
        for (p1, p2, same) in raw_pairs:
            # CSVの場合は絶対パスか相対パスかを判定して結合
            if not os.path.isabs(p1): p1 = os.path.join(images_dir, p1)
            if not os.path.isabs(p2): p2 = os.path.join(images_dir, p2)
            pairs.append((p1, p2, same))
    else:
        # USE FACTORY
        factory = PairsParserFactory(pairs_path, pairs_format)
        parser = factory.get_parser()
        if parser is None:
            raise ValueError(f"Unknown pairs format: {pairs_format}. Supported: LFW, CPLFW, CALFW, AgeDB30, RFW")
        
        print(f"Using parser for: {pairs_format}")
        try:
            raw_pair_list = parser.parse_pairs()
        except Exception as e:
            print(f"Error parsing pairs: {e}")
            raise

        # 2. Resolve Paths (Smart Extension Check)
        for (name1, name2, label) in raw_pair_list:
            # まずそのまま結合して存在チェック
            p1 = os.path.join(images_dir, name1)
            p2 = os.path.join(images_dir, name2)

            # 画像1の存在確認と拡張子スワップ
            if not os.path.exists(p1):
                # .png -> .jpg を試す
                if p1.endswith('.png'):
                    p1_alt = p1.replace('.png', '.jpg')
                    if os.path.exists(p1_alt):
                        p1 = p1_alt
                # .jpg -> .png を試す
                elif p1.endswith('.jpg'):
                    p1_alt = p1.replace('.jpg', '.png')
                    if os.path.exists(p1_alt):
                        p1 = p1_alt

            # 画像2の存在確認と拡張子スワップ
            if not os.path.exists(p2):
                if p2.endswith('.png'):
                    p2_alt = p2.replace('.png', '.jpg')
                    if os.path.exists(p2_alt):
                        p2 = p2_alt
                elif p2.endswith('.jpg'):
                    p2_alt = p2.replace('.jpg', '.png')
                    if os.path.exists(p2_alt):
                        p2 = p2_alt
            
            # Convert label to bool
            issame = True if int(label) == 1 else False
            pairs.append((p1, p2, issame))

    npairs = len(pairs)
    total_imgs = npairs * 2
    
    if npairs == 0:
        raise ValueError(f"No valid pairs found in {pairs_path} using format {pairs_format}")

    data_flips = []
    for flip in [0,1]:
        data_flips.append(torch.empty((total_imgs, 3, img_h, img_w), dtype=torch.float32))
    issame_list = []
    idx = 0

    print(f"Start loading {npairs} pairs ({total_imgs} images) using OpenCV...")
    
    for (p1, p2, same) in pairs:
        for p in (p1, p2):
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                # ここでエラーになる場合、パスが間違っているかファイルが破損している
                raise FileNotFoundError(f"OpenCV could not read image: {p}\nEnsure the file exists and the path is correct.")
            
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            h, w = img.shape[:2]
            target_short = image_size[0]
            
            if min(w, h) != target_short:
                scale = float(target_short) / float(min(w, h))
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
            data_flips[0][idx].copy_(t)
            
            img_flip = cv2.flip(img, 1)
            tflip = torch.from_numpy(img_flip.transpose(2, 0, 1).copy()).float()
            data_flips[1][idx].copy_(tflip)
            
            idx += 1
            
        issame_list.append(same)
        if idx % 1000 == 0:
            print('loaded images', idx)

    print("loaded total images:", idx)
    return data_flips, issame_list

@torch.no_grad()
def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f) 
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')
            
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]), dtype=torch.float32)
        data_list.append(data)
        
    print(f"Start loading .bin file: {path} ({len(bins)} images)...")
    
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img_bgr = cv2.imdecode(np.frombuffer(_bin, np.uint8), -1)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        if min(w, h) != image_size[0]:
            target_short = image_size[0]
            scale = float(target_short) / float(min(w, h))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
        t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        data_list[0][idx].copy_(t)
        
        img_flip = cv2.flip(img, 1)
        tflip = torch.from_numpy(img_flip.transpose(2, 0, 1).copy()).float()
        data_list[1][idx].copy_(tflip)
        
        if idx % 1000 == 0:
            print('loading bin', idx)

    print(data_list[0].shape)
    return data_list, issame_list

# ========= (PyTorch Inference Loop) =========

@torch.no_grad()
def test_pytorch(data_set, model, batch_size, device, nfolds=10):
    print('Testing verification (PyTorch)...')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    
    model.eval()
    total_time = 0.0

    for i in range(len(data_list)):
        data = data_list[i]
        n_samples = data.shape[0]
        embeddings = None
        ba = 0
        
        while ba < n_samples:
            bb = min(ba + batch_size, n_samples)
            count = bb - ba
            _data = data[ba:bb]
            img_tensor = ((_data / 255.0) - 0.5) / 0.5
            img_tensor = img_tensor.to(device)

            start_time = datetime.datetime.now()
            net_out = model(img_tensor)
            
            if isinstance(net_out, torch.Tensor):
                _embeddings = net_out.detach().cpu().numpy()
            else:
                _embeddings = np.array(net_out)
                
            end_time = datetime.datetime.now()
            total_time += (end_time - start_time).total_seconds()
            
            if embeddings is None:
                embeddings = np.zeros((n_samples, _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings
            ba = bb
            
        embeddings_list.append(embeddings)

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    
    print(f'Inference time: {total_time:.4f}s')
    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)
    return acc_mean, acc_std, val, val_std, far


# ========= (Main) =========

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Verification Test')
    
    parser.add_argument('--images-dir', type=str, default='', help='Root directory of images')
    parser.add_argument('--pairs', type=str, default='', help='Path to pairs.txt')
    parser.add_argument('--data-dir', type=str, default='', help='Directory for .bin files')
    parser.add_argument('--target', type=str, default='lfw', help='Targets for .bin loading')
    
    parser.add_argument('--model-prefix', type=str, default='', help='Path to model file (.pt)')
    parser.add_argument('--model-path', type=str, default='', help='Alias for model-prefix')
    
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--nfolds', type=int, default=10, help='Number of folds')
    # 重要な引数: LFW, CPLFW, CALFW, AgeDB30 などを指定
    parser.add_argument('--pairs-format', type=str, default='LFW', help='Parser format: LFW, CPLFW, CALFW, AgeDB30')
    parser.add_argument('--epochs', type=str, default='1', help='Ignored')

    args = parser.parse_args()

    model_file = args.model_path if args.model_path else args.model_prefix
    if not model_file:
        raise ValueError("Please provide a model path using --model-prefix")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_size = [112, 112]

    # --- 1. Load Model ---
    print(f"Loading model from {model_file} ...")
    try:
        loaded_obj = torch.load(model_file, map_location=device)
        # --- 正しい読み込みの例 ---
        if isinstance(loaded_obj, torch.nn.Module):
            model = loaded_obj
        elif isinstance(loaded_obj, dict):
            # loaded_obj がネイティブな state_dict か、'state_dict' に入っているかをチェック
            sd = loaded_obj
            if 'state_dict' in loaded_obj:
                sd = loaded_obj['state_dict']

            # モデル名をファイル名から推測
            if 'r50' in model_file.lower(): name = 'r50'
            elif 'r100' in model_file.lower(): name = 'r100'
            else: name = 'r50'

            print(f"Initializing backbone {name} (use SE) ...")
            # use_se=True を渡して SE を有効にしたバックボーンを作る
            model = get_model(name, fp16=False, use_se=True)

            # 場合によっては key に "module." がついているため取り除く
            new_sd = {}
            for k, v in sd.items():
                new_key = k.replace('module.', '')  # DataParallel 等に対応
                new_sd[new_key] = v
            model.load_state_dict(new_sd, strict=False)


        else:
            raise ValueError(f"Unknown model format")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    model.to(device)
    model.eval()

    # --- 2. Load Datasets ---
    datasets_to_test = []
    
    if args.images_dir and args.pairs:
        print(f"Mode: PNG Images from {args.images_dir}")
        print(f"Format: {args.pairs_format}")
        dataset_name = os.path.basename(args.pairs)
        # pairs_format を渡す
        ds = load_images_from_pairs(args.images_dir, args.pairs, image_size, args.pairs_format)
        datasets_to_test.append((dataset_name, ds))
        
    elif args.data_dir:
        print(f"Mode: BIN files from {args.data_dir}")
        targets = args.target.split(',')
        for name in targets:
            path = os.path.join(args.data_dir, name + ".bin")
            if os.path.exists(path):
                print(f"Loading {name}...")
                ds = load_bin(path, image_size)
                datasets_to_test.append((name, ds))
            else:
                print(f"Warning: {path} not found.")
    else:
        print("Error: Invalid arguments.")
        exit(1)

    # --- 3. Run Evaluation ---
    for name, data_set in datasets_to_test:
        acc, std, val, val_std, far = test_pytorch(data_set, model, args.batch_size, device, args.nfolds)
        print('\n' + '='*40)
        print(f'[{name}] Results:')
        print(f'  Accuracy: {acc:.5f} +- {std:.5f}')
        print(f'  Validation Rate: {val:.5f} +- {val_std:.5f} @ FAR={far:.5f}')
        print('='*40 + '\n')