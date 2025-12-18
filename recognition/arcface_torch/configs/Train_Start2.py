from easydict import EasyDict as edict

# torchrun --nproc_per_node=1 train_v2.py configs/Train_Start.py
# によって実行
config = edict()

# --- 1. 損失関数 (ArcFace) の設定 ---
config.margin_list = (1.0, 0.5, 0.0) # ArcFaceのマージン値 (m1, m2, m3)。m2(0.5)が重要。

# --- 2. モデル（バックボーン）の設定 ---
config.network = "r50"                              # バックボーン: "mbf"(軽量), "r50"(標準), "r100"(高精度)
config.head_name = "partial_fc"               # head名 (partial_fc,BoundaryFace,exBoundaryFace)
config.resume = False                               # 学習を再開するか (True/False)
config.output = "/workspace/Output/vface10k_r50_ArcFace"         # モデルとログの保存先 (例: "/workspace/runs/my_run")
config.embedding_size = 512                         # 顔の特徴ベクトルの次元数 (通常512)
config.sample_rate = 1.0                            # Partial FC のサンプリングレート (1.0は全クラス使用)
config.fp16 = True                                  # 半精度(FP16)学習で高速化・省メモリ化
config.use_se = True                              # SEブロックを使うか (True/False)

# --- BoundaryFace (BoundaryMargin) 専用パラメータの追加 ---
# losses.py内のBoundaryMarginクラスで参照されます。
config.boundary_s = 32.0                    # BoundaryFaceのスケールファクター s (デフォルト: 32.0)
config.boundary_m = 0.5                       # BoundaryFaceのマージン m (デフォルト: 0.5)
config.boundary_epoch_start = 30            # ノイズ除去処理を開始するエポック (デフォルト: 7)

# 新機能(OSN/CSN)用パラメータ
config.boundary_K_nn = 5         # boundary_K ではなく boundary_K_nn に変更
config.boundary_tau_osn = 0.15    # boundary_tau ではなく boundary_tau_osn に変更

# --------------------------------------------------------
# --- 3. 学習ハイパーパラメータ ---
config.momentum = 0.9                               # SGDオプティマイザのモメンタム値 (通常0.9)
config.weight_decay = 5e-4                          # 過学習を防ぐ重み減衰 (L2正則化)
config.batch_size = 256                            # 1GPUあたりのバッチサイズ (VRAMに応じて調整)
config.lr = 0.1                                     # 学習率 (Learning Rate) の初期値
config.verbose = 2000                               # 検証データセット（lfwなど）を実行する頻度 epoch ことにするなら:総画像枚数 / バッチサイズ = 1エポックあたりのステップ数
config.frequent = 100                               # LossやSpeedの進捗ログを表示する頻度
config.dali = False                                 # NVIDIA DALIローダーを使うか (.recならTrue, 画像フォルダならFalse)



# --- 4. データセットの設定 (vface10k 用に変更) ---

# ※ vface10k はリストファイル(.txt)と画像フォルダが別々のため、
#   ローダー(dataset.py)が両方のパスを必要とする可能性がある

"""
# (A) リストファイル (.txt) へのパス
config.rec = "/workspace/TrainDataSets/casia/meta/train.txt"

# (B) 画像のルートディレクトリ (ローダーが参照する場合に備えて定義)
#     (もし dataset.py が config.rec のパスから自動で推測する場合は不要)
config.data_root = "/workspace/TrainDataSets/casia/train"

#  casia の実際の値
config.num_classes = 10572             # ※データセットのクラス数 (＝人数)
config.num_image = 490624              # ※データセットの総画像枚数
"""


# (A) リストファイル (.txt) へのパス
config.rec = "/workspace/TrainDataSets/vface10k/meta/train.txt"

# (B) 画像のルートディレクトリ (ローダーが参照する場合に備えて定義)
#     (もし dataset.py が config.rec のパスから自動で推測する場合は不要)
config.data_root = "/workspace/TrainDataSets/vface10k/train"

# ※ vface10k の実際の値
config.num_classes = 10000             # ※データセットのクラス数 (＝人数)
config.num_image = 452455              # ※データセットの総画像枚数

# --- 5. 学習スケジュールの設定 ---
config.num_epoch = 40                  # 学習する総エポック数
config.warmup_epoch = 0                # 学習率ウォームアップのエポック数 (0はウォームアップなし)
config.val_targets = ['lfw', 'cfp_fp', "agedb_30", 'calfw' , 'cplfw'] # 学習中に検証するテストセット名 
config.scheduler = "Multistep"        # 学習率スケジューラの種類 ("Cosine", "Multistep", "Polynomial" から選択)
config.milestones = [18, 28, 35]        # MultiStepLR用のマイルストーンエポックリスト (schedulerが"Multistep"の場合に使用)
config.gamma = 0.1              # 変更時の減衰率


# --- 6. ClearML設定 ---
#task_name = f"{hostname}-{head_name}-{backbone}-{other_tags}"    のようになる
config.project_name = "Suzuki_project"          # プロジェクト名
config.host_name = "pkh3402c"           # サーバー名
config.backbone_name = "se-iResNet50"            # backbone名
config.other_tags = "vface10k"              # その他タグ

