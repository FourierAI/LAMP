#!/usr/bin/env python3
"""Prepare a local support bank from official dataset splits and raw images.

CoOp split JSON files contain train/val/test lists of
[path_relative_to_images, integer_label, class_name]. Aircraft and ImageNet
also support their native annotations. Class labels and annotation order are
preserved. ImageNet's native validation split is also its evaluation split.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import random

from PIL import Image
import torch
from torchvision import transforms as T

from dataset_specs import DATASET_TEMPLATES, SUPPORTED_DATASETS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
VIEWS = 10


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def seed_value(value):
    number = int(value)
    if not 0 <= number < 2**63:
        raise argparse.ArgumentTypeError("Seed must be between 0 and 2**63 - 1")
    return number


def relative_id(value):
    """Accept portable relative paths; reject traversal and absolute paths."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid relative image path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts or not path.parts:
        raise ValueError(f"Image paths must stay relative to --images: {value!r}")
    return path.as_posix()


def image_path(root, identity):
    path = (root / relative_id(identity)).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Image is not a file inside --images: {identity}")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"train", "val", "test"}:
        raise ValueError("Split JSON must have exactly the keys train, val, and test")
    return data


def aircraft_splits(images, annotations=None):
    directory = annotations or images.parent
    classes = [line.strip() for line in (directory / "variants.txt").read_text(encoding="utf-8").splitlines()]
    if not classes or any(not name for name in classes) or len(set(classes)) != len(classes):
        raise ValueError("variants.txt must contain one distinct, nonempty class name per line")
    labels = {name: label for label, name in enumerate(classes)}
    splits = {}
    for phase in ("train", "val", "test"):
        rows = []
        path = directory / f"images_variant_{phase}.txt"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2 or parts[1] not in labels:
                raise ValueError(f"Invalid Aircraft annotation at {path}:{number}")
            identity, name = parts
            rows.append([relative_id(identity + ".jpg"), labels[name], name])
        splits[phase] = rows
    return splits


def imagenet_splits(images, classnames=None):
    """Read train/<synset> and val/<synset>; classnames.txt sets label order."""
    source = classnames or images / "classnames.txt"
    names = {}
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Expected '<synset> <class name>' at {source}:{number}")
        synset, name = parts
        if relative_id(synset) != synset or "/" in synset or synset in names:
            raise ValueError(f"Invalid or duplicate synset at {source}:{number}")
        names[synset] = name
    if not names:
        raise ValueError("classnames.txt is empty")
    splits = {}
    for phase in ("train", "val"):
        directory = images / phase
        folders = {path.name for path in directory.iterdir() if path.is_dir() and not path.name.startswith(".")}
        if folders != set(names):
            raise ValueError(f"{directory} class folders must match classnames.txt exactly")
        rows = []
        for label, (synset, name) in enumerate(names.items()):
            paths = sorted(path for path in (directory / synset).rglob("*")
                           if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
            if not paths:
                raise ValueError(f"No images in {directory / synset}")
            rows.extend([[path.relative_to(images).as_posix(), label, name] for path in paths])
        splits[phase] = rows
    splits["test"] = [row.copy() for row in splits["val"]]
    return splits


def validate_splits(data, dataset, shot):
    """Validate identities and use numeric labels to preserve class order."""
    splits, classnames, labels_by_id = {}, {}, {}
    for phase in ("train", "val", "test"):
        raw_rows = data[phase]
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError(f"{phase} must contain at least one image")
        rows, seen = [], set()
        for row in raw_rows:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ValueError(f"{phase} rows must be [relative_path, label, class_name]")
            identity = relative_id(row[0])
            label, name = row[1:]
            if type(label) is not int or label < 0 or not isinstance(name, str) or not name.strip():
                raise ValueError(f"Invalid class label/name in {phase}: {row!r}")
            if identity in seen:
                raise ValueError(f"Duplicate image in {phase}: {identity}")
            seen.add(identity)
            if label in classnames and classnames[label] != name:
                raise ValueError(f"Class name differs for label {label}")
            if identity in labels_by_id and labels_by_id[identity] != label:
                raise ValueError(f"Image has conflicting labels: {identity}")
            classnames[label] = name
            labels_by_id[identity] = label
            rows.append([identity, label, name])
        splits[phase] = rows
    if set(classnames) != set(range(len(classnames))):
        raise ValueError("Class labels must be contiguous integers starting at zero")
    identities = {phase: {row[0] for row in rows} for phase, rows in splits.items()}
    if identities["train"] & (identities["val"] | identities["test"]):
        raise ValueError("Training images overlap validation or test images")
    overlap = bool(identities["val"] & identities["test"])
    if overlap and (dataset != "imagenet" or identities["val"] != identities["test"]):
        raise ValueError("Validation/test overlap is allowed only for identical ImageNet query sets")
    for phase, rows in splits.items():
        counts = defaultdict(int)
        for _, label, _ in rows:
            counts[label] += 1
        minimum = shot if phase == "train" else 1
        if any(counts[label] < minimum for label in classnames):
            raise ValueError(f"{phase} needs at least {minimum} image(s) for every class")
    return splits, [classnames[label] for label in range(len(classnames))]


def sample_support(train, class_count, shot, seed):
    pools = defaultdict(list)
    for row in train:
        pools[row[1]].append(row)
    rng = random.Random(seed)
    return [row for label in range(class_count) for row in rng.sample(pools[label], shot)]


def select_validation(rows, class_count, cap, seed):
    if cap is None:
        return rows
    pools = defaultdict(list)
    for row in rows:
        pools[row[1]].append(row)
    rng = random.Random(seed)
    selected = {row[0] for label in range(class_count)
                for row in rng.sample(pools[label], min(cap, len(pools[label])))}
    return [row for row in rows if row[0] in selected]


def support_geometry(ids, sizes, seed):
    geometry = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for _ in range(VIEWS):
            for identity in ids:
                width, height = sizes[identity]
                with Image.new("RGB", (width, height)) as image:
                    top, left, crop_h, crop_w = T.RandomResizedCrop.get_params(image, (.5, 1.), (.75, 4 / 3))
                flip = int(torch.rand(()) < .5)
                geometry.append([width, height, top, left, crop_h, crop_w, flip])
    return geometry


def prompt_tokens(classes, templates):
    try:
        import clip
    except ImportError as error:
        raise RuntimeError("Install the official OpenAI CLIP package listed in requirements.txt") from error
    if not callable(getattr(clip, "tokenize", None)):
        raise RuntimeError("The installed 'clip' module is not OpenAI CLIP; install requirements.txt")
    return [clip.tokenize([template.format(name.replace("_", " ")) for template in templates]).tolist()
            for name in classes]


def prepare(args):
    root = args.images.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--images must be a directory")
    if args.output.exists():
        raise FileExistsError(f"Output already exists; choose a new bank path: {args.output}")
    if args.split_json is not None:
        raw = read_split_json(args.split_json)
    elif args.dataset == "fgvc":
        raw = aircraft_splits(root, args.annotations)
    elif args.dataset == "imagenet":
        raw = imagenet_splits(root, args.classnames)
    else:
        raise ValueError(f"--split-json is required for {args.dataset}")
    splits, classes = validate_splits(raw, args.dataset, args.shot)
    support = sample_support(splits["train"], len(classes), args.shot, args.seed)
    validation = select_validation(splits["val"], len(classes), args.val_per_class, args.validation_seed)
    test = splits["test"]
    templates = DATASET_TEMPLATES[args.dataset]
    tokens = prompt_tokens(classes, templates)
    ids = [row[0] for row in support]
    set_ids = set(ids)
    required = sorted(set_ids | {row[0] for row in validation} | {row[0] for row in test})
    hashes, sizes, resolved = {}, {}, {}
    for number, identity in enumerate(required, 1):
        path = image_path(root, identity)
        if path in resolved and resolved[path] != identity:
            raise ValueError(f"Image aliases have different identities: {resolved[path]}, {identity}")
        resolved[path] = identity
        hashes[identity] = sha256(path)
        if identity in set_ids:
            with Image.open(path) as image:
                if image.width <= 0 or image.height <= 0:
                    raise ValueError(f"Invalid image dimensions: {identity}")
                sizes[identity] = image.size
        if number % 10000 == 0:
            print(f"Prepared {number}/{len(required)} image hashes", flush=True)
    geometry = support_geometry(ids, sizes, args.seed)
    overlap = bool({row[0] for row in validation} & {row[0] for row in test})
    bank = {
        "classes": classes,
        "templates": list(templates),
        "prompt_token_ids": tokens,
        "validation": validation,
        "test": test,
        "image_sha256": hashes,
        "support_banks": {f"{args.shot}_{args.seed}": {
            "ids": ids, "labels": [row[1] for row in support], "geometry": geometry,
        }},
        "selection_evaluation_overlap": overlap,
        "preparation": {
            "dataset": args.dataset, "shot": args.shot, "seed": args.seed,
            "views": VIEWS, "training_rows": len(splits["train"]),
            "validation_seed": args.validation_seed, "val_per_class": args.val_per_class,
            "full_validation_rows": len(splits["val"]),
            "support_sampling": "Python random.Random(seed).sample; numeric class order",
            "geometry_sampling": "torch.manual_seed(seed); view-major crop then flip",
        },
    }
    payload = json.dumps(bank, separators=(",", ":"), allow_nan=False).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        with gzip.GzipFile(fileobj=stream, filename="", mode="wb", mtime=0) as compressed:
            compressed.write(payload)
    print(f"Wrote {args.output}: {len(classes)} classes, {len(ids)} support images / "
          f"{len(geometry)} views, {len(validation)} validation and {len(test)} test images")
    if overlap:
        print("ImageNet selection and evaluation share image identities.")
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--images", required=True, type=Path, help="Root for all relative image paths")
    parser.add_argument("--split-json", type=Path, help="CoOp train/val/test JSON; optional for native Aircraft/ImageNet")
    parser.add_argument("--shot", required=True, type=positive_int, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--seed", required=True, type=seed_value)
    parser.add_argument("--output", required=True, type=Path, help="New local bank.json.gz path")
    parser.add_argument("--val-per-class", type=positive_int, help="Optional cap; test split remains complete")
    parser.add_argument("--validation-seed", type=seed_value, default=14001)
    parser.add_argument("--annotations", type=Path, help="Aircraft annotation directory; default: parent of --images")
    parser.add_argument("--classnames", type=Path, help="ImageNet synset/name file; default: --images/classnames.txt")
    args = parser.parse_args()
    if args.annotations is not None and (args.dataset != "fgvc" or args.split_json is not None):
        parser.error("--annotations is used only for native Aircraft input")
    if args.classnames is not None and (args.dataset != "imagenet" or args.split_json is not None):
        parser.error("--classnames is used only for native ImageNet input")
    try:
        prepare(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
