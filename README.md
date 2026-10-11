# LAMP

**LAMP: Layerwise Adaptive Mixing of Predictors for Training-Free Few-Shot Vision–Language Adaptation**

Zhipeng Ye, Feng Jiang, Qiufeng Wang, and Hao Li

LAMP adapts a frozen CLIP model by fitting global and native-layer kernel predictors to shared support-label residuals, then mixing their class scores. It preserves each layer's feature dimension and uses no gradient optimization. The optional LAMP-F extension adapts text-anchored class prototypes with an exact cached kernel response. Both methods use labeled validation data to select their predictors.

This implementation supports **ViT-B/16** and **RN50** on eleven classification datasets.

## Installation

Tested with Python 3.12, PyTorch 2.11, and torchvision 0.26. A CUDA GPU is required for fitting and inference.

```bash
git clone https://github.com/FourierAI/LAMP.git
cd LAMP
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Obtain the corresponding [OpenAI CLIP checkpoint](https://github.com/openai/CLIP) and pass its local path with `--checkpoint`.

## Data preparation

Follow the [CoOp dataset instructions](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to download the images and benchmark splits. For example, DTD should contain:

```text
/data/dtd/
├── images/
│   ├── banded/
│   └── ...
└── split_zhou_DescribableTextures.json
```

Create a support bank with four training images per class and ten augmentation views per support image:

```bash
python prepare_data.py \
  --dataset dtd \
  --images /data/dtd/images \
  --split-json /data/dtd/split_zhou_DescribableTextures.json \
  --shot 4 --seed 1 \
  --output cache/dtd_4_1.json.gz
```

Keep this bank file to reuse the same support identities and augmentation views. The support budget counts training images; validation examples are additional.

The CoOp JSON loader supports `caltech101`, `oxford_pets`, `stanford_cars`, `oxford_flowers`, `food101`, `sun397`, `dtd`, `eurosat`, and `ucf101`. Set `--images` to the directory against which the split's image paths resolve.

Two datasets also have native loaders:

- **FGVC Aircraft (`fgvc`):** set `--images` to the dataset's `data/images` directory. The parent directory supplies `variants.txt` and `images_variant_{train,val,test}.txt`; use `--annotations` for another annotation directory.
- **ImageNet (`imagenet`):** set `--images` to a directory containing `train/<synset>/` and `val/<synset>/`, and provide `--classnames /data/imagenet/classnames.txt`. Each class-name line contains a synset ID followed by its class name.

For these native loaders, omit `--split-json` and retain the same `--shot`, `--seed`, and `--output` options. Use `python prepare_data.py --help` for all preparation options.

## Run LAMP

```bash
python main.py \
  --dataset dtd \
  --backbone ViT-B/16 \
  --shot 4 --seed 1 \
  --images /data/dtd/images \
  --checkpoint /models/ViT-B-16.pt \
  --bank-file cache/dtd_4_1.json.gz \
  --output outputs/dtd/vit_b16/4shot/seed1 \
  --method lamp
```

The run also evaluates the paired global-only predictor. Replace `--method lamp` with `--method both` to include LAMP-F. For RN50, use `--backbone RN50` and its corresponding checkpoint. Supported shot counts are `1`, `2`, `4`, `8`, and `16`.

By default, the command runs verification, feature encoding, parameter selection, optional prototype adaptation, and evaluation. Intermediate stages can be selected with `--stage`; see `python main.py --help`.

Dense kernel fitting uses quadratic storage and cubic factorization work in the number of augmented support entries. ImageNet at 16 shots has 160,000 entries and requires approximately 191 GiB of scratch storage for the dense solver. Start with a smaller dataset such as DTD when configuring the environment.

## Citation

```bibtex
@misc{ye2026lamp,
  title = {LAMP: Layerwise Adaptive Mixing of Predictors for Training-Free Few-Shot Vision-Language Adaptation},
  author = {Ye, Zhipeng and Jiang, Feng and Wang, Qiufeng and Li, Hao},
  year = {2026},
  note = {Manuscript}
}
```

## Acknowledgements

This implementation builds on [CLIP](https://github.com/openai/CLIP) and [ProKeR](https://github.com/ybendou/ProKeR), with benchmark conventions from [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter) and [CoOp](https://github.com/KaiyangZhou/CoOp).
