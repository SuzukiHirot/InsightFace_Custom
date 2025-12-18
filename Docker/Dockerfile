# 
# NVIDIA NGC コンテナ
# CUDA 11.8, PyTorch 1.14.0a0 相当が含まれています
FROM nvcr.io/nvidia/pytorch:22.12-py3

#
# 1. System Libraries
# 基本ツールとOpenCVに必要なライブラリ群
#
RUN apt-get update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev tzdata fonts-ipafont \
    vim \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/local/src/*
ENV TZ=Asia/Tokyo

#
# 2. Python Setup
#
RUN pip3 install --upgrade pip

#
# 3. Install Main Libraries
# ここではバージョンを指定せず、依存パッケージごとインストールさせます。
# ※ この段階では新しいNumpy(1.24+)が入ってしまいますが、次のステップで修正します。
#
RUN pip3 install --no-cache-dir \
    clearml==1.13.1 \
    boto3 \
    Cython \
    scikit-learn \
    scikit-image \
    matplotlib \
    prettytable \
    easydict \
    albumentations==1.4.18 \
    onnx \
    onnxruntime-gpu \
    insightface \
    pyyaml \
    scipy \
    tensorboard \
    termcolor \
    timm==0.4.12 \
    yacs

#
# 4. Version Fix (CRITICAL STEP)
# エラー回避のため、NumpyとOpenCVを「動作確認済みのバージョン」に強制変更します。
# 依存関係チェックを行わないオプション(--no-deps)をつけることで、
# albumentations等の競合エラーによるビルド停止を防ぎます。
#
RUN pip3 install --no-cache-dir --force-reinstall --no-deps \
    numpy==1.23.5 \
    opencv-python-headless==4.7.0.72

# 作業ディレクトリの設定
WORKDIR /workspace

# コンテナ起動時にbashを実行
CMD ["/bin/bash"]