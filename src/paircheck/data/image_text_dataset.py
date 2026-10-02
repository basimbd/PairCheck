from PIL import Image
from torch.utils.data import Dataset


class ImageTextDataset(Dataset):
    def __init__(self, image_paths: list[str], captions: list[str], labels: list[int], preprocessor):
        self.image_paths = image_paths
        self.captions = captions
        self.labels = labels

        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "image": self.preprocessor(Image.open(self.image_paths[idx])),
            "caption": self.captions[idx],
            "label": self.labels[idx]
        }
