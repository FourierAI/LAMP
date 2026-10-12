# LAMP

**LAMP: Layerwise Adaptive Mixing of Predictors for Training-Free Few-Shot Vision–Language Adaptation**

Zhipeng Ye, Feng Jiang, Qiufeng Wang, and Hao Li

LAMP adapts a frozen CLIP model by fitting global and native-layer kernel predictors to shared support-label residuals, then mixing their class scores. It preserves each layer's feature dimension and uses no gradient optimization. The optional LAMP-F extension adapts text-anchored class prototypes with an exact cached kernel response. Both methods use labeled validation data to select their predictors.

This implementation supports **ViT-B/16** and **RN50** on eleven classification datasets.

## Installation

Use Python 3.12 with the pinned dependencies below. A CUDA GPU is required for fitting and inference. The recorded experiments used PyTorch 2.11 and torchvision 0.26; the historical replay instructions below also specify the image decoder.

```bash
git clone https://github.com/FourierAI/LAMP.git
cd LAMP
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the original [OpenAI CLIP checkpoints](https://github.com/openai/CLIP):

```bash
mkdir -p checkpoints
curl -fL --retry 3 \
  https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt \
  -o checkpoints/ViT-B-16.pt
curl -fL --retry 3 \
  https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt \
  -o checkpoints/RN50.pt
```

`main.py` verifies their SHA-256 hashes against `configs/checkpoints.json`. Use these original TorchScript `.pt` files, rather than converted state dictionaries or safetensors.

## Data preparation

Follow the [CoOp dataset instructions](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to download the images and benchmark splits. For example, DTD should contain:

```text
/data/dtd/
├── images/
│   ├── banded/
│   └── ...
└── split_zhou_DescribableTextures.json
```

Create a fresh support bank with four training images per class and ten augmentation views per support image:

```bash
python prepare_data.py \
  --dataset dtd \
  --images /data/dtd/images \
  --split-json /data/dtd/split_zhou_DescribableTextures.json \
  --shot 4 --seed 1 \
  --output cache/dtd_4_1.json.gz
```

Keep this bank file to reuse the same support identities and augmentation views. `prepare_data.py` samples new identities and crop geometry from the supplied split; using a paper seed alone does not reconstruct the paper's saved bank. The support budget counts training images; validation examples are additional.

The CoOp JSON loader supports `caltech101`, `oxford_pets`, `stanford_cars`, `oxford_flowers`, `food101`, `sun397`, `dtd`, `eurosat`, and `ucf101`. Set `--images` to the directory against which the split's image paths resolve.

Two datasets also have native loaders:

- **FGVC Aircraft (`fgvc`):** set `--images` to the dataset's `data/images` directory. The parent directory supplies `variants.txt` and `images_variant_{train,val,test}.txt`; use `--annotations` for another annotation directory.
- **ImageNet (`imagenet`):** set `--images` to a directory containing `train/<synset>/` and `val/<synset>/`, and provide `--classnames /data/imagenet/classnames.txt`. Each class-name line contains a synset ID followed by its class name.

For these native loaders, omit `--split-json` and retain the same `--shot`, `--seed`, and `--output` options. Use `python prepare_data.py --help` for all preparation options.
The native ImageNet loader uses the official validation images for both selection and evaluation; the manifest and result record this overlap.

## Run LAMP

```bash
python main.py \
  --dataset dtd \
  --backbone ViT-B/16 \
  --shot 4 --seed 1 \
  --images /data/dtd/images \
  --checkpoint checkpoints/ViT-B-16.pt \
  --bank-file cache/dtd_4_1.json.gz \
  --output outputs/dtd/vit_b16/4shot/seed1 \
  --method lamp
```

The run also evaluates the paired global-only predictor. Replace `--method lamp` with `--method both` to include LAMP-F. For RN50, use `--backbone RN50` and its corresponding checkpoint. Supported shot counts are `1`, `2`, `4`, `8`, and `16`.

By default, the command runs verification, feature encoding, parameter selection, optional prototype adaptation, and evaluation. Intermediate stages can be selected with `--stage`; see `python main.py --help`. Encoding and later stages record the numerical environment, image decoder, and input/source hashes in `runtime.json`. Resuming requires the same inputs, code, batch/solver settings, and numerical environment. Outputs from older code without decoder records need either their original code/environment or a new output directory.

To add LAMP-F after a completed LAMP selection, reuse the bank and output directory with `--method both --stage adapt`, then `--method both --stage evaluate`. Adaptation creates `prototype_response.pt` and `prototype_response.json` for the selected active branches; evaluation reuses them. Training-free LAMP skips this preparation.

Dense kernel fitting uses quadratic storage and cubic factorization work in the number of augmented support entries. ImageNet at 16 shots has 160,000 entries and requires approximately 191 GiB of scratch storage for the dense solver. Start with a smaller dataset such as DTD when configuring the environment.

## Reproducing the reported experiments

The manuscript submission includes a separate `Reproducibility.zip`; it is not currently hosted as a public download. The archive contains frozen input manifests, the original executable, and experiment records. Paths in this section are relative to its extracted root. The preserved directory name `PRISM_v14` refers to LAMP/LAMP-F.

### Frozen support banks

Use a saved manifest to reuse the paper's exact support identities, ten-view crop geometry, prompts, and query lists. For example, from this repository:

```bash
python main.py \
  --dataset dtd --backbone ViT-B/16 --shot 1 --seed 1 \
  --images /data/dtd/images \
  --checkpoint checkpoints/ViT-B-16.pt \
  --bank-file /path/to/Reproducibility/PRISM_v14/datasets/dtd.json.gz \
  --output outputs/dtd/frozen_1shot_seed1 \
  --method both
```

Each `PRISM_v14/datasets/<dataset>.json.gz` contains the recorded banks for shots 1/2/4/8/16 and seeds 1/2/3. Reusing these inputs is distinct from numerical replay of the archived executable.

For historical replay, use the original `PRISM_v14/run.py`. Its `environment.yml` and `binary_packages.json` specify Python 3.12.9, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, and Pillow 11.1.0 linked to JPEG 9f. A pip Pillow version alone does not fix the JPEG decoder. Prepare that recorded environment, check decoded pixels, and run into a fresh output directory:

```bash
conda env create -n lamp-replay \
  -f /path/to/Reproducibility/PRISM_v14/environment.yml
conda activate lamp-replay
python /path/to/Reproducibility/PRISM_v14/check_decoder.py \
  --dataset dtd --images /data/dtd/images
python /path/to/Reproducibility/PRISM_v14/run.py \
  --dataset dtd --backbone ViT-B/16 --shot 1 --seed 1 \
  --images /data/dtd/images \
  --checkpoint checkpoints/ViT-B-16.pt \
  --bank-file /path/to/Reproducibility/PRISM_v14/datasets/dtd.json.gz \
  --output outputs/dtd/archived_1shot_seed1 \
  --method both --device cuda:0
```

The environment recipe records the runtime used for the original checks; a fresh installation has not been validated end to end. `PRISM_v14/HISTORICAL_COMPARISON.json` records the checked configurations and agreement with saved predictions.

### Experiment entry points and records

The public `main.py` supports ViT-B/16 and RN50. Additional-backbone and frozen-transfer rows below point to archived results and protocols, rather than public inference commands.

| Experiment | Entry point or archive path | Scope |
|---|---|---|
| Main LAMP/LAMP-F benchmarks | Public `main.py`; frozen banks under `PRISM_v14/datasets/` | Eleven datasets, five shot counts, three recorded seeds; original executable: `PRISM_v14/run.py`. |
| Matched representation controls | `evidence/reproducibility/replay_representation.py` | Replays stored predictions for the ten-dataset, four-shot REGION/CENTROID/MULTIBAND comparisons. |
| New support draws and validation budgets | `evidence/supplementary_experiment_artifacts/v14_validation_budget/REPRODUCE.txt` | 120 configurations on DTD/EuroSAT; `verify_outputs.py` and `report.py` replay predictions/statistics, and the instructions give raw-image execution with frozen bank files. |
| RN101 | `evidence/tables/v4_transfer_rn101_dataset_results.csv`; `evidence/tables/v4_transfer_rn101_aggregates.csv` | Training-free LAMP: eleven datasets, five shots, three seeds; archived summaries. |
| ViT-B/32 | `evidence/tables/backbone_comparison.csv`; `evidence/tables/v4_transfer_cell_provenance.csv` | Training-free LAMP: ImageNet, 16 shots, three seeds; archived results and provenance. |
| Frozen ImageNet-V2/Sketch transfer | `evidence/tables/v4_transfer_ood_per_seed.csv`; `evidence/tables/v4_transfer_ood_summary.csv` | ViT-B/16 and RN50 LAMP, 16-shot source models, three seeds; source parameters remain fixed. |
| 20,000 additional ImageNet queries | `evidence/supplementary_experiment_artifacts/v14_unused_imagenet/REPRODUCE.txt` | `checks/verify_outputs.py` verifies stored predictions; full forward replay requires the source models/features identified in that directory's `SOURCE_LOCK.json`. |

`evidence/enriched_backbones_transfer_appendix.tex` describes the additional-backbone and transfer protocols. Original images, dataset annotations, and CLIP checkpoints are obtained from their respective providers.

## Checks

Run the CPU regression checks for response caching, selected-model integrity, and resumed environments:

```bash
python -m unittest discover -s tests
```

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
