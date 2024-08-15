import os
import tempfile

import torchvision

from tqdm.auto import trange


CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def main():
    for split in ["train", "test"]:
        out_dir = os.path.join("/data/Dynamics/cifar_data", f"cifar_{split}")
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print(f"downloading to {tmp_dir}...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.CIFAR10(root=tmp_dir, train=split == "train", download=True)

        print(f"dumping images to {out_dir}...")
        os.mkdir(out_dir)
        for i in trange(len(dataset)):
            image, label = dataset[i]
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            image.save(filename)


if __name__ == "__main__":
    main()
