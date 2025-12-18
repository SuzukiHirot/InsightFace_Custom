python eval/test_PNG.py \
  --images-dir /workspace/TestDataSets/test/LFW/val \
  --pairs /workspace/TestDataSets/test/LFW/pairs.txt \
  --pairs-format lfw \
  --model-prefix /workspace/model_epoch_40.pt \
  --epochs 1 \
  --batch-size 32 \
  --gpu 0

python eval/test_PNG.py \
  --images-dir /workspace/TestDataSets/test/CALFW/val \
  --pairs /workspace/TestDataSets/test/CALFW/pairs.txt \
  --pairs-format calfw \
  --model-prefix /workspace/model_epoch_40.pt \
  --epochs 1 \
  --batch-size 32 \
  --gpu 0

python eval/test_PNG.py \
  --images-dir /workspace/TestDataSets/test/CPLFW/val \
  --pairs /workspace/TestDataSets/test/CPLFW/pairs.txt \
  --pairs-format cplfw \
  --model-prefix /workspace/model_epoch_40.pt \
  --epochs 1 \
  --batch-size 32 \
  --gpu 0
