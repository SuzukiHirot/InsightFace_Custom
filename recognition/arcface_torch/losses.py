import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import os

# -----------------------------------------------------------------------------
# InsightFace Original Losses
# -----------------------------------------------------------------------------

class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s        

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits


class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.4):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits


# -----------------------------------------------------------------------------
# Boundary Margin Implementations
# -----------------------------------------------------------------------------


class BoundaryMargin(nn.Module):
    """
    BoundaryFace (paper-faithful, interface-compatible)

    forward(x, label, epoch, img_path, out_dir)
    return: output, final(boundary loss), rectified_label
    """

    def __init__(
        self,
        in_feature=128,
        out_feature=10575,
        s=32.0,
        m=0.4,
        easy_margin=False,
        epoch_start=25
    ):
        super().__init__()

        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.epoch_start = epoch_start
        self.easy_margin = easy_margin

        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label, epoch, img_path, out_dir):
        """
        Args:
            x: feature (B, F)
            label: GT label (B,)
            epoch: current epoch
            img_path: list[str]
            out_dir: str
        """

        # -------------------------
        # cosine(theta)
        # -------------------------
        cosine = F.linear(
            F.normalize(x),
            F.normalize(self.weight)
        )

        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        rectified_label = label
        final = torch.zeros(1, device=x.device)

        # ======================================================
        # BoundaryFace logic (epoch >= epoch_start)
        # ======================================================
        if epoch >= self.epoch_start:
            with torch.no_grad():
                # GT cosine
                gt_cos = torch.sum(one_hot * cosine, dim=1)

                # impostor cosine
                impostor_cos = cosine * (1.0 - one_hot)
                max_impostor_cos, max_impostor_idx = impostor_cos.max(dim=1)

                # Closed-set noise判定
                csn_mask = max_impostor_cos > gt_cos

                # ラベル修正
                rectified_label = torch.where(
                    csn_mask,
                    max_impostor_idx,
                    label
                )

                # ログ出力（元コード互換）
                if img_path is not None and out_dir is not None:
                    os.makedirs(out_dir, exist_ok=True)
                    file_path = os.path.join(out_dir, f'BoundaryTXT{epoch}.txt')

                    with open(file_path, 'a') as f:
                        for i in torch.nonzero(csn_mask).squeeze(1):
                            path_i = img_path[i]
                            f.write(
                                f"{path_i}\t{rectified_label[i].item()}\n"
                            )

            # -------------------------
            # Boundary loss (hard samples)
            # -------------------------
            margin_gap = max_impostor_cos - gt_cos

            # 境界付近（越えてはいない）
            hard_mask = (margin_gap < 0) & (margin_gap > -self.m)

            if hard_mask.any():
                final = torch.mean(
                    (-margin_gap[hard_mask])
                ) * math.pi

        one_hot_final = torch.zeros_like(cosine)
        one_hot_final.scatter_(1, rectified_label.view(-1, 1), 1.0)

        output = (
            one_hot_final * phi +
            (1.0 - one_hot_final) * cosine
        )
        output = output * self.s

        return output, final, rectified_label


class exBoundaryMargin(nn.Module): 
    # tau_osn (OSN判定閾値)をデフォルト引数に追加
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.40, easy_margin=False, epoch_start=30, K_nn=3, tau_osn=0.25):
        super(exBoundaryMargin, self).__init__() 
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        # クラス中心ベクトル W
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # ArcFaceのモノトニック条件
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.epoch_start = epoch_start
        
        # 新規追加のハイパーパラメータ
        self.K_nn = K_nn             # クラス中心KNNで参照する近傍クラス数 K
        self.tau_osn = tau_osn       # OSN判定用の類似度閾値 (Open-Set Noise Threshold)


    def forward(self, x, label, epoch, img_path, out_dir):
        # 1. 類似度の計算 (ArcFace 標準)
        # cos(theta): xと全クラス中心Wのコサイン類似度
        norm_x = F.normalize(x)
        norm_w = F.normalize(self.weight)
        cosine = F.linear(norm_x, norm_w)

        # cos(theta + m): 正解クラスに適用するマージン付きコサイン類似度
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # easy_margin or L-Softmax like monotonic condition
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # 閾値以下の領域にペナルティを加える（ArcFaceの標準的な処理）
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # One-Hotエンコーディング: マージン適用とノイズ修正のベース
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # ノイズ修正ロジックを実行しない初期期間
        if epoch <= self.epoch_start:
            rectified_label = label
            final = torch.tensor(0.0).to(x.device)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s
            return output, final, rectified_label
        
        # -----------------------------------------------
        # epoch > self.epoch_start: KNNベースのノイズ修正パイプライン
        # -----------------------------------------------
        
        # ノイズ判定ロジックからの勾配逆伝播を阻止するために detach() を使用
        cosine_detached = cosine.detach() 

        # -----------------------------------------------
        # STAGE 1: Open-Set Noise (OSN) の判定と棄却
        # -----------------------------------------------
        
        # Top-Kの類似度を取得
        # K_nnの数がクラス総数を超える可能性を考慮して min を取る
        K_val = min(self.K_nn, self.out_feature)
        topk_results = torch.topk(cosine_detached, K_val, dim=1)
        topk_scores = topk_results.values # [B, K_val]

        # Top-Kスコアの平均値を計算 (OSN判定基準)
        avg_topk_score = torch.mean(topk_scores, dim=1) # [B]
        
        # OSN判定: 平均スコアが閾値未満のサンプルをOSNと判定
        is_inlier = avg_topk_score >= self.tau_osn
        
        # OSNとして棄却されたサンプルのインデックスを取得
        osn_indices = torch.nonzero(~is_inlier).squeeze(1)
        
        # ログの出力: OSN 棄却
        if osn_indices.numel() > 0 and img_path is not None:
            osn_original_labels = label[osn_indices].cpu().numpy()
            try:
                # ファイル名を OSN_Rejected_{epoch}.txt に統一
                file_path = os.path.join(out_dir, f'OSN_Rejected_{epoch}.txt')
                with open(file_path, 'a') as f:
                    for i, original_idx in enumerate(osn_indices):
                        # 画像パスと元のラベルを出力 (これらのサンプルは損失計算で無視されるべき)
                        current_path = img_path[original_idx] if isinstance(img_path, (list, tuple)) else str(img_path[original_idx])
                        f.write(f'{current_path}\t{osn_original_labels[i]}\n')
            except IOError:
                pass

        # inlier (ノイズではない/修正対象) サンプルのみのインデックスを取得
        inlier_indices = torch.nonzero(is_inlier).squeeze(1)
        
        # -----------------------------------------------
        # STAGE 2: Closed-Set Noise (CSN) の判定と修正 (Inlierのみ)
        # -----------------------------------------------
        
        # インライアが一つもない場合は、早期終了 (ただし、上記OSNログは記録済み)
        if inlier_indices.numel() == 0:
            rectified_label = label
            final = torch.tensor(0.0).to(x.device)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s
            return output, final, rectified_label
        
        # 1. Inlierサンプルのコサイン類似度とラベルを取得
        inlier_cosine = cosine_detached[inlier_indices] # [B_inlier, C]
        inlier_label = label[inlier_indices]          # [B_inlier]
        
        # 2. Inlierに対して Top-K を計算
        inlier_topk_results = torch.topk(inlier_cosine, K_val, dim=1)
        inlier_topk_indices = inlier_topk_results.indices # [B_inlier, K_val]
        
        # 3. CSN判定: Top-Kのクラスインデックスに、真のラベルが含まれているかチェック
        # Top-K内に含まれていないサンプルを CSN と判定
        is_correct_in_topk = (inlier_topk_indices == inlier_label.unsqueeze(1)).any(dim=1) # [B_inlier]
        
        # CSNと判定されたサンプルの、inlier内でのインデックスを取得
        csn_inlier_indices = torch.nonzero(~is_correct_in_topk).squeeze(1)
        
        # CSNサンプルの元のバッチインデックスを特定
        csn_original_indices = inlier_indices[csn_inlier_indices]

        # 4. ラベルの修正とログ出力 (CSNと判定された場合)
        if csn_original_indices.numel() > 0:
            # Top-1で最も近いクラスを新しいラベルとする (修正後のラベル)
            new_labels = inlier_topk_indices[csn_inlier_indices, 0] 
            
            # CSN修正ログの出力
            if img_path is not None:
                try:
                    file_path = os.path.join(out_dir, f'CSN_Corrected_{epoch}.txt')
                    with open(file_path, 'a') as f:
                        for i, original_idx in enumerate(csn_original_indices):
                            new_label_val = new_labels[i].item()
                            original_label_val = label[original_idx].item()
                            
                            # one_hot ベクトルを修正 (損失計算に使用)
                            one_hot_line = torch.zeros((1, self.out_feature)).to(x.device)
                            one_hot_line.scatter_(1, new_labels[i].unsqueeze(0).unsqueeze(-1), 1)
                            one_hot[original_idx] = one_hot_line.squeeze(0)

                            # 画像パス、元のラベル、修正後のラベルを出力
                            current_path = img_path[original_idx] if isinstance(img_path, (list, tuple)) else str(img_path[original_idx])
                            f.write(f'{current_path}\t{original_label_val}\t{new_label_val}\n')
                except IOError:
                    pass

        # -----------------------------------------------
        # 3. 最終的なロジット出力と補助損失の計算
        # -----------------------------------------------
        
        # 修正後の one_hot を使って、マージン付き Softmax のロジットを計算
        right = one_hot * phi      # 正解クラスの位置にはマージン付きスコア phi
        left = (1.0 - one_hot) * cosine # 不正解クラスの位置にはマージンなしスコア cosine
        output = left + right
        output = output * self.s

        # 補助損失 final の計算 (BoundaryFaceのロジックを流用し、inlierのみ考慮)
        # OSNサンプルは損失計算から除外されるため、inlier_indicesでマスク
        masked_left = left[inlier_indices]
        masked_right = right[inlier_indices]
        
        # Inlierに対してのみマージン違反をチェック (決定境界安定化項)
        left_max, _ = torch.max(masked_left, dim=1)  
        right_max, _ = torch.max(masked_right, dim=1) 
        
        sub = left_max - right_max
        zero = torch.zeros_like(sub)
        temp = torch.where(sub > 0, sub, zero)
        
        # 補助損失 L_boundary
        final = torch.mean(temp) * math.pi
        
        # 最終的な修正済みラベル (one_hotから抽出)
        rectified_label = torch.topk(one_hot, 1)[1].squeeze(1)

        return output, final, rectified_label