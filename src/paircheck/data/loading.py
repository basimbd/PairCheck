import math
import random
from PIL import Image
from torch.utils.data import Dataset, Sampler


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


class BalancedBatchSampler(Sampler):
    def __init__(self, pos_indices, neg_indices, batch_size: int):
        self.pos_indices = list(pos_indices)
        self.neg_indices = list(neg_indices)

        self.total_samples = len(self.pos_indices) + len(self.neg_indices)
        self.total_batches = math.ceil(self.total_samples / batch_size)

        self.batch_size = batch_size
        self.pos_batch_size = batch_size // 2
        self.neg_batch_size = (batch_size - self.pos_batch_size)

    def __get_batch(self, items: list, idx: int, batch_size: int):
        batch = [
            items[(idx + offset) % len(items)]  # ensures wrapping to the front of the list if length is exceeded
            for offset in range(batch_size) 
        ]
        next_idx = (idx + batch_size) % len(items)
        return batch, next_idx

    def __iter__(self):
        pos_pool = self.pos_indices.copy()
        neg_pool = self.neg_indices.copy()
        random.shuffle(pos_pool)
        random.shuffle(neg_pool)
        pos_idx = 0
        neg_idx = 0
        for _ in range(self.total_batches):
            pos_batch, pos_idx = self.__get_batch(pos_pool, pos_idx, self.pos_batch_size)
            neg_batch, neg_idx = self.__get_batch(neg_pool, neg_idx, self.neg_batch_size)
            batch = pos_batch + neg_batch
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.total_batches


if __name__ == "__main__":
    # Test the BalancedBatchSampler wrapping
    # with batch_size > len(dataset) case.
    pos_indices = list(range(5))
    neg_indices = list(range(5, 10))
    batch_size = 32
    sampler = BalancedBatchSampler(pos_indices, neg_indices, batch_size)
    for batch in sampler:
        print(len(batch))
