"""Federated dataset class for enabling pre-fetching of single clients' datasets."""

from collections.abc import Callable
from logging import INFO
import time
from torch.utils.data import Dataset, DataLoader

from flwr.common import log

from pollen_worker.pollen_utils import get_client_ds_fn, get_clients_population_dict


class FederatedDataset(Dataset):
    """Open Image dataset object."""

    def __init__(
        self,
        dataset_generator: Callable[[int], Dataset],
        list_of_clients: list[int] | list[str],
        verbose: bool = False,
    ) -> None:
        self.dataset_generator = dataset_generator
        self.list_of_clients = list_of_clients
        self.verbose = verbose

    def __getitem__(self, index: int) -> Dataset:
        """Return the client dataset with client id `index`."""
        if self.verbose:
            log(INFO, f"Getting client {index} dataset.")
        return self.dataset_generator(int(self.list_of_clients[index]))

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.list_of_clients)


# class DataSampler(torch.utils.data.Sampler):

#     def __init__(self, underlying_iterator):
#         self._underlying_iterator = underlying_iterator

#     def __iter__(self):
#         return self

#     def __next__(self):
#         return next(self._underlying_iterator)


if __name__ == "__main__":
    """Testing over all the dataset types."""
    # Loop over all the dataset types
    for dataset_name, batch_size in zip(
        [
            "shakespeare_memory",
            "flair",
            "openimage",
            "shakespeare",
            "google_speech",
            "reddit",
        ],
        [11, 1, 20, 11, 20, 20],
        strict=False,
    ):

        def _dataset_generator(cid: int, _ds_name: str = dataset_name) -> Dataset:
            ds, _ = get_client_ds_fn(name=_ds_name)(cid)
            return ds

        # Test the dataset
        n_clients_to_sample = 100
        cid_samples_dict = get_clients_population_dict(
            name=dataset_name,
            batch_size=batch_size,
            seed=51550,
        )
        list_of_cids = [int(x) for x in cid_samples_dict][:n_clients_to_sample]
        dataset = FederatedDataset(
            dataset_generator=_dataset_generator,
            list_of_clients=list_of_cids,
            verbose=True,
        )
        assert len(dataset) == len(list_of_cids)
        dataset_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x[0],
            prefetch_factor=5,
            num_workers=1,
        )
        for loaded_ds in dataset_loader:
            time.sleep(1.0)
            log(INFO, f"Returned dataset for client {loaded_ds.client_id}")
