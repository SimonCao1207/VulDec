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


def get_embedding(model, tokenizer, code):
    input_ids = tokenizer(
        code, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    )["input_ids"]
    with torch.no_grad():
        context_embeddings = model(input_ids)[0].squeeze(0)
        # print(context_embeddings.shape)
    return context_embeddings


class CodeBlockDataset(Dataset):
    def __init__(self, model, tokenizer, dataset):
        self.samples = dataset
        self.model = model
        self.tokenizer = tokenizer
        os.makedirs("data/embeddings", exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code, label = self.samples[idx]
        code_embedding = get_embedding(model, tokenizer, code)
        return code_embedding, label


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


def get_data():
    print("Loading data...")
    train_set = pickle.load(open(f"data/{mode}_train_set.pkl", "rb"))
    val_set = pickle.load(open(f"data/{mode}_val_set.pkl", "rb"))
    test_set = pickle.load(open(f"data/{mode}_test_set.pkl", "rb"))

    # Create torch Dataset and DataLoader
    train_dataset = CodeBlockDataset(model, tokenizer, train_set)
    val_dataset = CodeBlockDataset(model, tokenizer, val_set)
    test_dataset = CodeBlockDataset(model, tokenizer, test_set)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader


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

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    # Create torch Dataset and DataLoader
    train_dataset = CodeBlockDataset(model, tokenizer, train_set)
    val_dataset = CodeBlockDataset(model, tokenizer, val_set)
    test_dataset = CodeBlockDataset(model, tokenizer, test_set)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Example usage:
    for batch_x, batch_y in train_loader:
        # batch_x: [batch_size, ...], batch_y: [batch_size]
        print(batch_x[0])
        print("label:", batch_y[0])
        break
