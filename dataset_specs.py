"""Text prompt templates used by the eleven LAMP benchmarks."""

DATASET_TEMPLATES = {
    "caltech101": ("a photo of a {}.",),
    "oxford_pets": ("a photo of a {}, a type of pet.",),
    "stanford_cars": ("a photo of a {}.",),
    "oxford_flowers": ("a photo of a {}, a type of flower.",),
    "food101": ("a photo of {}, a type of food.",),
    "fgvc": ("a photo of a {}, a type of aircraft.",),
    "sun397": ("a photo of a {}.",),
    "dtd": ("{} texture.",),
    "eurosat": ("a centered satellite photo of {}.",),
    "ucf101": ("a photo of a person doing {}.",),
    "imagenet": (
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ),
}

SUPPORTED_DATASETS = tuple(DATASET_TEMPLATES)
