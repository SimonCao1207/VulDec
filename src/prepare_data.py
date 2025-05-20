import json
import os
import pickle
import random

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# import torch
# from transformers import AutoModel, AutoTokenizer
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


def get_embedding(code):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    code_tokens = tokenizer.tokenize(code)
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.eos_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
    return context_embeddings


class CodeBlockDataset(Dataset):
    def __init__(self, dataset, name):
        self.X, self.Y = [], []
        print(f"Creating {name} dataset... ({mode})")
        for code, label in dataset:
            code_embedding = get_embedding(code)
            self.X.append(code_embedding)
            self.Y.append(label)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = torch.tensor(self.Y[idx])
        return x, y


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

    print(f"Total samples: {n}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Create torch Dataset and DataLoader
    train_dataset = CodeBlockDataset(train_set, "training")
    val_dataset = CodeBlockDataset(val_set, "validation")
    test_dataset = CodeBlockDataset(test_set, "test")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Example usage:
    for batch_x, batch_y in train_loader:
        # batch_x: [batch_size, ...], batch_y: [batch_size]
        print(batch_x[0])
        print("label:", batch_y[0])
        break
