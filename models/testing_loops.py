from typing import List, Tuple

import hydra
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AlbertTokenizer
from transformers.modeling_outputs import MaskedLMOutput
import wandb
import yaml

transformers.logging.set_verbosity_error()

from datasets.nlp_util import mask_tokens
from utils import wandb_init
from pollen_utils import get_client_ds


def get_testing_loop(name: str):
    if name == "reddit":
        return reddit_testing_loop
    elif name == "google_speech":
        return google_speech_testing_loop
    else:
        return general_testing_loop


def reddit_testing_loop(
    testloader: DataLoader,
    device: torch.device,
    net: Module,
    tokenizer: AlbertTokenizer,
    **kwargs,
):
    test_loss = 0.0
    test_len = 0
    num_masked = 0
    num_correct = 0

    net.eval()
    with torch.no_grad():
        for data in tqdm(testloader):
            try:
                data: torch.Tensor = data.to(device=device)
                data, target, masked_indices = mask_tokens(
                    data, tokenizer, mlm_probability=0.15, device=device
                )
                target = target.to(device=device)
                num_masked += len(target[masked_indices])

                output: MaskedLMOutput = net(input_ids=data, labels=target)
                test_loss += output.loss.item()
                predictions = output.logits.max(2)[1]
                # Only computing accuracy on the masked tokens
                num_correct += (
                    (predictions[masked_indices] == data[masked_indices])
                    .clone()
                    .detach()
                    .sum()
                    .item()
                )

            except Exception as ex:
                print(f"Testing failed as {ex}")
                break
            test_len += len(target)

        test_len = max(test_len, 1)
        # Test loss averages over number of batches
        test_loss /= len(testloader)
        test_loss = round(test_loss, 4)
        # Accuracy averages over number of masked tokens
        accuracy = round(num_correct / num_masked, 4)
        test_metrics = {"accuracy": accuracy}

    return test_loss, test_len, test_metrics


def google_speech_testing_loop(
    testloader: DataLoader,
    device: torch.device,
    net: Module,
    criterion: Module,
    **kwargs,
):
    test_loss = 0
    test_len = 0
    num_correct = 0

    net.eval()
    with torch.no_grad():
        for data, target in tqdm(testloader):
            try:
                data: torch.Tensor = data.to(device=device)
                data = torch.unsqueeze(data, 1)
                target: torch.Tensor = target.to(device=device)
                test_len += len(target)

                output: torch.Tensor = net(data)
                num_correct += (
                    (output.max(1)[1] == target).clone().detach().sum().item()
                )

                loss: torch.Tensor = criterion(output, target)
                test_loss += loss.item()
            except Exception as ex:
                print(f"Testing failed as {ex}")
                break

        # Number of test samples
        test_len = max(test_len, 1)
        # Test loss averages over number of batches
        test_loss /= len(testloader)
        test_loss = round(test_loss, 4)
        # Accuracy averages over number of samples
        accuracy = round(num_correct / test_len, 4)

        test_metrics = {
            "accuracy": accuracy,
        }

    return test_loss, test_len, test_metrics


def general_testing_loop(
    testloader: DataLoader,
    device: torch.device,
    net: Module,
    criterion: Module,
    **kwargs,
):
    test_loss = 0
    test_len = 0
    num_correct = 0

    net.eval()
    with torch.no_grad():
        for data, target in tqdm(testloader):
            try:
                data: torch.Tensor = data.to(device=device)
                target: torch.Tensor = target.to(device=device)
                test_len += len(target)

                output: torch.Tensor = net(data)
                num_correct += (
                    (output.max(1)[1] == target).clone().detach().sum().item()
                )
                loss: torch.Tensor = criterion(output, target)
                test_loss += loss.item()
            except Exception as ex:
                print(f"Testing failed as {ex}")
                break

        # Number of test samples
        test_len = max(test_len, 1)
        # Test loss averages over number of batches
        test_loss /= len(testloader)
        test_loss = round(test_loss, 4)
        # Accuracy averages over number of samples
        accuracy = round(num_correct / test_len, 4)

        test_metrics = {
            "accuracy": accuracy,
        }

    return test_loss, test_len, test_metrics


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)
) -> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred: torch.Tensor = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        return res


def tmp_fn(name, cids, dataset):
    clients_test_sets = []
    for cid in cids:
        ds, tokenizer = get_client_ds(name=name, cid=cid, dataset=dataset)
        clients_test_sets.append(ds)
    return clients_test_sets, tokenizer


@hydra.main(config_path="../conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python models/testing_loops.py output_dir="/nfs-share/ls985/pollen_worker/outputs/2023-09-06/10-59-14" task="reddit"
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python models/testing_loops.py output_dir="/nfs-share/ls985/pollen_worker/outputs/2023-09-05/18-53-45" task="google_speech"
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python models/testing_loops.py output_dir="/nfs-share/ls985/pollen_worker/outputs/2023-09-05/18-53-11" task="openimage"
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python models/testing_loops.py output_dir="/nfs-share/ls985/pollen_worker/outputs/2023-09-05/18-52-53" task="shakespeare_memory"
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python models/testing_loops.py output_dir="/nfs-share/ls985/pollen_worker/outputs/2023-09-05/18-52-57" task="shakespeare_memory"
    import pickle
    from collections import OrderedDict
    from logging import INFO
    from multiprocessing import Pool
    from pathlib import Path

    import numpy as np
    import psutil
    from flwr.common import parameters_to_ndarrays
    from flwr.common.logger import log
    from flwr.common.typing import NDArrays
    from torch.utils.data import ConcatDataset
    from transformers import AlbertForMaskedLM

    from datasets.nlp_util import get_collate_fn
    from pollen_utils import get_clients_population_dict, get_device, get_model

    # device = "cpu"
    device = get_device()

    def set_parameters(parameters: NDArrays, net: Module = None, device=device):
        if net is None:
            net = get_model(name=cfg.task.name)
        net.eval()
        keys = [k for k in net.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v, device=device) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=False)
        return net

    def chunks_idx(l, n):
        d, r = divmod(len(l), n)
        for i in range(n):
            si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
            yield si, si + (d + 1 if i < r else d)

    log(
        INFO,
        f"Offline evaluation of task {cfg.task.name}. Using output_dir: {cfg.output_dir}",
    )
    # Set the root directory
    root_dir = Path(cfg.output_dir)
    # Get test_loop fn
    test_loop = get_testing_loop(name=cfg.task.name)
    # Get the list of cids
    cid_samples_dict = get_clients_population_dict(
        name=cfg.task.name,
        batch_size=1,
        dataset="test",
    )
    # Get clients' test sets
    n_jobs = 100
    try:
        cpus = len(psutil.Process().cpu_affinity())
    except AttributeError:
        cpus = psutil.cpu_count()
    if n_jobs > cpus:
        n_jobs = cpus
    clients_test_sets = []
    pool_inputs = []
    pool = Pool(n_jobs)
    client_ids = list(cid_samples_dict.keys())[:10]
    for begin, end in chunks_idx(range(len(client_ids)), n_jobs):
        pool_inputs.append([cfg.task.name, client_ids[begin:end], "test"])
    pool_outputs = pool.starmap(tmp_fn, pool_inputs)
    pool.close()
    pool.join()
    [[clients_test_sets.append(a) for a in x[0]] for x in pool_outputs]
    tokenizer = pool_outputs[0][1]
    # Concatenate the clients test sets
    testset = ConcatDataset(clients_test_sets)
    log(INFO, f"Test set size: {len(testset)}")
    # Instantiate the test loader
    # NOTE: This batch sizes are estimated to fill the VRAM
    # or maximise the utilisations of a single 2080
    batch_sizes = {
        "reddit": 256,
        "google_speech": 512,
        "openimage": 1024,
        "shakespeare_memory": 256,
    }
    testloader = DataLoader(
        testset,
        batch_size=batch_sizes[cfg.task.name],
        shuffle=False,
        pin_memory=True,
        collate_fn=get_collate_fn(tokenizer=tokenizer)
        if tokenizer is not None
        else None,
    )
    with open (root_dir / ".hydra"/"config.yaml", "r") as f:
        wandb_config = yaml.safe_load(f)

    # Create results .csv file
    results_file = root_dir / "offline_eval_results.csv"
    net = None
    # Get the models' performance
    with wandb_init(
        cfg.use_wandb,
        **cfg.wandb.setup,
        settings=wandb.Settings(start_method="thread"),
        config=wandb_config,  # type: ignore
    ) as run:
        for i, parameters_file in enumerate(root_dir.glob("parameters_aggregated_*")):
            round = int(parameters_file.name.split("_")[-1])
            with open(parameters_file, "rb") as f:
                parameters: NDArrays = pickle.load(f)
            net = set_parameters(
                parameters=parameters_to_ndarrays(parameters), net=net, device=device
            )
            net.to(device=device)
            net.eval()
            criterion = torch.nn.CrossEntropyLoss(reduction="mean").to(device=device)
            test_res = test_loop(
                testloader=testloader,
                device=device,
                net=net,
                tokenizer=tokenizer,
                criterion=criterion,
            )
            if i == 0:
                with open(results_file, "w") as f:
                    metrics = ",".join([f"{k}" for k, _ in test_res[2].items()])
                    f.write(f"round,test_loss,{metrics}\n")
                log(
                    INFO,
                    f"A file containing the round number, the average test loss, and metrics ({metrics}) will be written",
                )
            wandb.log({f"test_loss": test_res[0], **test_res[2]}, step=round)
            with open(results_file, "a") as f:
                metrics = ",".join([f"{v}" for _, v in test_res[2].items()])
                f.write(f"{round},{test_res[0]},{metrics}\n")
            log(INFO, f"Round {round}, test loss: {test_res[0]}, metrics: {metrics}")
            break


if __name__ == "__main__":
    main()
