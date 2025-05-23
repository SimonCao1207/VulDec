import json
import os
import pickle
import random

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from myutils import findposition, findpositions, getblocks

mode = "sql"
random.seed(42)


def load_data(mode="sql"):
    """
    scheme :
    {
        "github_link": {
            "commit_hash": {
                "url": <github_link_to_commit>,
                "htlml_url": <github_link_to_html>,
                "sha": <sha>,
                "keywords": <example: sql injection prevent>,
                "diff": <diff_content>,
                "files": {
                    "file_name": {
                        "changes": [
                            {
                                "diff" : <diff_content>,
                                "badparts": [<list of bad parts>],
                                "goodparts": [<list of good parts>],
                            },
                            ...
                        ],
                    }
                },
                "msg": <commit_msg>,
            }

        }
    }
    """

    with open("data/plain_" + mode, "r") as infile:
        data = json.load(infile)
    return data


class CodeBlockDataset(Dataset):
    def __init__(self, tokenizer, dataset):
        self.samples = dataset
        self.tokenizer = tokenizer
        os.makedirs("data/embeddings", exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code, label = self.samples[idx]
        inputs = self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def snapshot_data(data):
    """
    snapshot a random commit
    """
    import random

    links = list(data.keys())
    if links:
        link = random.choice(links)
        commit_hashes = list(data[link].keys())
        if commit_hashes:
            commit_hash = random.choice(commit_hashes)
            print(f"files : {data[link][commit_hash]['files']}")
            print(data[link][commit_hash]["diff"])


def extract_blocks_from_data(data, step=5, fulllength=200):
    """
    Extract code blocks containing bad parts from the dataset.
    Returns a list of blocks.
    """
    count = 0
    allblocks = []
    for link in tqdm(data):
        for commit_hash in data[link]:
            if "files" in data[link][commit_hash]:
                for f in data[link][commit_hash]["files"]:
                    if "source" not in data[link][commit_hash]["files"][f]:
                        print("no source code in commit, skipping")
                        continue
                    if "source" in data[link][commit_hash]["files"][f]:
                        sourcecode = data[link][commit_hash]["files"][f]["source"]
                        allbadparts = []
                        for change in data[link][commit_hash]["files"][f]["changes"]:
                            badparts = change["badparts"]
                            count = count + len(badparts)
                            for bad in badparts:
                                # check if they can be found within the file
                                pos = findposition(bad, sourcecode)
                                if -1 not in pos:
                                    allbadparts.append(bad)
                        if len(allbadparts) > 0:
                            positions = findpositions(allbadparts, sourcecode)
                            blocks = getblocks(sourcecode, positions, step, fulllength)
                            for b in blocks:
                                allblocks.append(b)
            else:
                print("no files in detected")
    return allblocks


def prepare_dataloaders(tokenizer, mode="sql"):
    print("Preparing Data Loaders...")
    train_set = pickle.load(open(f"data/{mode}_train_set.pkl", "rb"))
    val_set = pickle.load(open(f"data/{mode}_val_set.pkl", "rb"))
    test_set = pickle.load(open(f"data/{mode}_test_set.pkl", "rb"))

    # Create torch Dataset and DataLoader
    train_dataset = CodeBlockDataset(tokenizer, train_set)
    val_dataset = CodeBlockDataset(tokenizer, val_set)
    test_dataset = CodeBlockDataset(tokenizer, test_set)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader


def load_model_and_tokenizer(path="microsoft/codebert-base"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    return model, tokenizer


if __name__ == "__main__":
    data = load_data()
    # snapshot_data(data)
    # exit()

    step = 5  # step length n in the description
    fulllength = 200  # context length m in the description
    if os.path.exists(f"data/{mode}_train_set.pkl"):
        print("Loading existing data...")
        with open(f"data/{mode}_train_set.pkl", "rb") as f:
            train_set = pickle.load(f)
        with open(f"data/{mode}_val_set.pkl", "rb") as f:
            val_set = pickle.load(f)
        with open(f"data/{mode}_test_set.pkl", "rb") as f:
            test_set = pickle.load(f)
    else:
        allblocks = extract_blocks_from_data(data, step=step, fulllength=fulllength)

        # Shuffle and split allblocks into train, val, test sets
        random.shuffle(allblocks)
        n = len(allblocks)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        n_test = n - n_train - n_val

        train_set = allblocks[:n_train]
        val_set = allblocks[n_train : n_train + n_val]
        test_set = allblocks[n_train + n_val :]

        # Save the splits for later loading
        with open(f"data/{mode}_train_set.pkl", "wb") as f:
            pickle.dump(train_set, f)
        with open(f"data/{mode}_val_set.pkl", "wb") as f:
            pickle.dump(val_set, f)
        with open(f"data/{mode}_test_set.pkl", "wb") as f:
            pickle.dump(test_set, f)

    print(f"Total samples: {len(train_set) + len(val_set) + len(test_set)}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    model, tokenizer = load_model_and_tokenizer()

    # Create torch Dataset and DataLoader
    train_dataset = CodeBlockDataset(tokenizer, train_set)
    val_dataset = CodeBlockDataset(tokenizer, val_set)
    test_dataset = CodeBlockDataset(tokenizer, test_set)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Example usage:
    for batch in train_loader:
        print(batch["input_ids"].shape)
        print("label:", batch["labels"][0])
        break
