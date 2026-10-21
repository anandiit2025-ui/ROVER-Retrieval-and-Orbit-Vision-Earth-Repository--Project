import json
import re
import zipfile
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset


REPO = "akshaydudhane/EarthDial-Dataset"
SETS = {
    "dubai": ("ChangeDetection_Shards/DUBAICC/DUBAI_ChangeDetection_train", 4),
    "levir": ("ChangeDetection_Shards/LEVIRCC/LEVIR_ChangeDetection_train", 4),
    "xbd": ("Temporal/Disaster_Shards/xBD/train/xBD_train_change_quest", 2),
}


def download(name, folder):
    subfolder, count = SETS[name]
    base = f"training_set/{subfolder}/"
    files = ["dataset_info.json", "state.json"]
    files += [f"data-{i:05d}-of-{count:05d}.arrow" for i in range(count)]
    for filename in files:
        hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=base + filename,
            local_dir=str(folder),
        )
    return Path(folder) / base


def clean_question(text):
    text = re.sub(r"\[[^]]+\]|<image>", " ", text)
    return " ".join(text.replace("\\n", " ").split())


def download_fpcd(folder):
    folder = Path(folder)
    files = ["T0.zip", "T1.zip", "multi_class_masks.zip",
             "cd_dataset_train.txt", "cd_dataset_test.txt"]
    for name in files:
        path = hf_hub_download(
            repo_id="ctundia/FPCD", repo_type="dataset",
            filename=name, local_dir=str(folder),
        )
        if name.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                archive.extractall(folder)
    return folder


class FPCDPairs(Dataset):
    def __init__(self, folder, limit=0, validation=False):
        folder = Path(folder)
        split = "test" if validation else "train"
        before = {}
        after = {}
        for name, items in [("T0", before), ("T1", after)]:
            for path in (folder / name).glob("*.jpg"):
                match = re.match(r"(.+)_\d{6}_(\d+)\.jpg$", path.name)
                if match:
                    items[f"{match[1]}_{match[2]}"] = path
        self.rows = []
        for mask in sorted((folder / "multi_class_masks" / split).glob("*.png")):
            key = mask.stem
            if key in before and key in after:
                self.rows.append((before[key], after[key], mask))
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        before, after, path = self.rows[i]
        with Image.open(path) as image:
            labels = {value for _, value in image.getcolors(maxcolors=256)}
        names = {1: "constructed", 2: "demolished", 3: "dried", 4: "filled with water"}
        changes = [names[value] for value in names if value in labels]
        answer = "Annotated farm-pond changes: " + ", ".join(changes) + "." if changes else "No farm-pond change is annotated in this pair."
        return {
            "before": Image.open(before).convert("RGB"),
            "after": Image.open(after).convert("RGB"),
            "question": "What changed in the farm ponds between these two images?",
            "answer": answer,
            "key": path.stem,
        }


class ChangePairs(Dataset):
    def __init__(self, path, limit=0, seed=42, validation=False):
        data = load_from_disk(str(path)).shuffle(seed=seed)
        if limit:
            data = data.select(range(min(limit, len(data))))
        count = max(1, int(len(data) * 0.1))
        self.data = data.select(range(count)) if validation else data.select(range(count, len(data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data[i]
        conversation = json.loads(row["conversations"])
        question = next(item["value"] for item in conversation if item["from"] == "human")
        answer = next(item["value"] for item in conversation if item["from"] == "gpt")
        return {
            "before": row["jpg_A"].convert("RGB"),
            "after": row["jpg_B"].convert("RGB"),
            "question": clean_question(question),
            "answer": answer.strip(),
            "key": row.get("__key__", str(i)),
        }
