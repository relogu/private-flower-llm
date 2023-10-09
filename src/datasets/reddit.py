import glob
import json
import os
import os.path
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AlbertTokenizer

from datasets.nlp_util import load_and_cache_examples

REDDIT_DTYPES = {
    "client_id": np.int64,
    "sample_path": "string",
    "label_name": np.int64,
    "label_id": np.int64,
}


class Reddit(Dataset):
    classes = []
    MAX_SEQ_LEN = 20000

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True):
        self.train = train  # training set or test set
        self.root = root

        self.data_file = "train" if train else "test"
        self.train = train

        self.vocab_tokens_size = 10000
        self.vocab_tags_size = 500

        # load data and targets
        self.raw_data, self.dict = self.load_file(self.root, self.train)

        if not self.train:
            self.raw_data = self.raw_data[:100000]
        else:
            self.raw_data = self.raw_data[:10000000]

        # we can't enumerate the raw data, thus generating artificial data to cheat the divide_data_loader
        self.data = [-1, len(self.dict)]
        self.targets = [-1, len(self.dict)]

    def __getitem__(self, index):
        """
        Args:xx
            index (int): Index
        Returns:
            tuple: (text, tags).
        """
        # Lookup tensor

        tokens = self.raw_data[index]
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = F.one_hot(tokens, self.vocab_tokens_size).float()
        tokens = tokens.mean(0)

        # tags = torch.tensor(tags, dtype=torch.long)
        # tags = F.one_hot(tags, self.vocab_tags_size).float()
        # tags = tags.sum(0)

        return tokens

    def __mapping_dict__(self):
        return self.dict

    def __len__(self):
        return len(self.raw_data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.data_file))

    def load_token_vocab(self, vocab_size, path):
        tokens_file = "reddit_vocab.pkl"
        with open(os.path.join(path, tokens_file), "rb") as f:
            tokens = pickle.load(f)
        return tokens[:vocab_size]

    def load_file(self, path, is_train):
        file_name = (
            os.path.join(path, "train") if self.train else os.path.join(path, "test")
        )

        # check whether we have generated the cache file before
        cache_path = (
            os.path.join(path, "train_cache")
            if self.train
            else os.path.join(path, "test_cache")
        )

        text = []
        mapping_dict = {}

        if os.path.exists(cache_path):
            print("====Load {} from cache".format(file_name))
            # dump the cache
            with open(cache_path, "rb") as fin:
                text = pickle.load(fin)
                mapping_dict = pickle.load(fin)
        else:
            print("====Load {} from scratch".format(file_name))
            # Mapping from sample id to target tag
            # First, get the token and tag dict
            vocab_tokens = self.load_token_vocab(self.vocab_tokens_size, path)

            vocab_tokens_dict = {k: v for v, k in enumerate(vocab_tokens)}
            client_data_list = []
            # Load the traning/testing data
            if self.train:
                data_files = sorted(glob.glob(file_name + "/*.json"))

                for f in data_files[:2]:
                    print("========Loading {}=========".format(f))
                    with open(f, "rb") as cin:
                        data = json.load(cin)
                    client_data_list.append(data)

            else:
                with open(os.path.join(file_name, "test.json"), "rb") as cin:
                    data = json.load(cin)
                client_data_list.append(data)

            count = 0
            clientCount = 0
            start_time = time.time()

            for client_data in client_data_list:
                client_list = client_data["users"]

                for clientId, client in enumerate(client_list):
                    tokens_list = list(client_data["user_data"][client]["x"])

                    for tokens in tokens_list:
                        tokens_list = [
                            vocab_tokens_dict[s]
                            for s in (tokens.split())
                            if s in vocab_tokens_dict
                        ]
                        if not tokens_list:
                            continue

                        mapping_dict[count] = clientId
                        text.append(tokens_list)

                        count += 1

                    clientCount += 1

                    num_of_remains = 1628176 - int(client)
                    print(
                        "====In loading data, remains {} clients, may take {} sec".format(
                            num_of_remains,
                            (time.time() - start_time) / clientCount * num_of_remains,
                        )
                    )
                    if clientId % 5000 == 0:
                        # dump the cache
                        with open(cache_path, "wb") as fout:
                            pickle.dump(text, fout)
                            pickle.dump(mapping_dict, fout)

                        print("====Dump for {} clients".format(clientId))

            # dump the cache
            with open(cache_path, "wb") as fout:
                pickle.dump(text, fout)
                pickle.dump(mapping_dict, fout)

        return text, mapping_dict


if __name__ == "__main__":
    data_root = Path("/datasets/FedScale/reddit/reddit")
    # train_dataset = Reddit(data_root)
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", do_lower_case=True)
    train_dataset = load_and_cache_examples(
        "albert-base-v2", data_root, tokenizer, n_jobs=11, evaluate=False
    )
    # test_dataset = load_and_cache_examples(
    #     'albert-base-v2',
    #     data_root,
    #     tokenizer,
    #     n_jobs=11,
    #     evaluate=True)

    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=4
    )

    for example in train_dataloader:
        print(example)
