# GPU training (Colab T4 or similar)

The repo was developed on a **CPU-only** machine. Everything below is the GPU path; the
numbers in `EVALUATION.md` come from the CPU runs described there, **not** from these
commands. Treat the steps marked *untested here* as a recipe, not a verified result.

## 0. Setup

```bash
git clone https://github.com/<you>/planread && cd planread
apt-get -qq install fonts-noto-cjk          # Noto Sans/Serif CJK JP (adds a Mincho/serif style we lack on Windows)
pip install -r requirements.txt
python scripts/download_models.py
python -m src.charset build
```

## 1. Data at full scale

```bash
python -m data_gen.make_datasets --what rec,orient,det --scale 5     # 200k train lines
python -m data_gen.make_docs --n 50
```

## 2. CRNN (in-repo recogniser)

```bash
python -m src.crnn train --device cuda --epochs 30 --batch 256 --lr 3e-3 --workers 4 --val-limit 2000
python -m eval.evaluate_rec --engines crnn
```

## 3. DBNet fine-tuning on crops with lines crossing text  *(untested here)*

`data/det/{train,test}/label.txt` is already in PaddleOCR's detection label format
(`path\t[{"transcription": ..., "points": [[x,y]*4]}]`).

```bash
pip install paddlepaddle-gpu paddleocr paddle2onnx
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR && cd PaddleOCR
# start from the official PP-OCRv5 mobile det config + pretrained weights, override the data paths
python tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml -o \
  Global.pretrained_model=<PP-OCRv5_mobile_det_pretrained.pdparams> \
  Global.epoch_num=50 Global.save_model_dir=../models/det_ft \
  Train.dataset.data_dir=../data/det/train Train.dataset.label_file_list=[../data/det/train/label.txt] \
  Eval.dataset.data_dir=../data/det/test Eval.dataset.label_file_list=[../data/det/test/label.txt]
python tools/export_model.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
  -o Global.pretrained_model=../models/det_ft/best_accuracy Global.save_inference_dir=../models/det_ft/infer
paddle2onnx --model_dir ../models/det_ft/infer --model_filename inference.json \
  --params_filename inference.pdiparams --save_file ../models/ppocr/det_finetuned.onnx
```

Then point `src/detect_lines.py::DET_MODEL` (or pass `path=`) at `det_finetuned.onnx` and
re-run `python -m eval.evaluate_det`.

## 4. Recogniser fine-tuning with PaddleOCR / PARSeq  *(not implemented)*

The `Recognizer` interface (`src/recognize.py`) accepts any CTC model that returns T×C
probabilities. Fine-tuning PP-OCR's recogniser on `data/rec` (it needs a PaddleOCR
`rec_gt.txt`: `path\ttext`, which is `labels.tsv` minus two columns) or adding PARSeq
(attention decoder: top-k would come from its beam search instead of CTC) are the next
steps; neither has been run.
