import json
import random

from tqdm import tqdm

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


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    # model = AutoModel.from_pretrained("microsoft/codebert-base")
    # nl_tokens = tokenizer.tokenize("return maximum value")
    # code_tokens = tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
    # tokens = (
    #     [tokenizer.cls_token]
    #     + nl_tokens
    #     + [tokenizer.sep_token]
    #     + code_tokens
    #     + [tokenizer.eos_token]
    # )
    # tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    # context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
    # print(context_embeddings)

    data = load_data()

    # Select a random link and a random commit_hash
    # import random

    # links = list(data.keys())
    # if links:
    #     link = random.choice(links)
    #     commit_hashes = list(data[link].keys())
    #     if commit_hashes:
    #         commit_hash = random.choice(commit_hashes)
    #         print(f"files : {data[link][commit_hash]['files']}")
    #         print(data[link][commit_hash]["diff"])  # print first 10 lines as a snapshot

    # exit()
    progress = 0
    count = 0
    step = 5  # step lenght n in the description
    fulllength = 200  # context length m in the description
    allblocks = []

    for link in tqdm(data):
        progress = progress + 1
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

    # Shuffle and split allblocks into train, val, test sets
    random.shuffle(allblocks)
    n = len(allblocks)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_set = allblocks[:n_train]
    val_set = allblocks[n_train : n_train + n_val]
    test_set = allblocks[n_train + n_val :]

    print(f"Total samples: {n}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Save the splits for later loading
    with open(f"data/{mode}_train_set.json", "w") as f:
        json.dump(train_set, f)
    with open(f"data/{mode}_val_set.json", "w") as f:
        json.dump(val_set, f)
    with open(f"data/{mode}_test_set.json", "w") as f:
        json.dump(test_set, f)
