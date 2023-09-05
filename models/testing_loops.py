from typing import List, Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch.autograd import Variable
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AlbertTokenizer

from datasets.nlp_util import mask_tokens
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
    test_loss = 0
    correct = 0
    top_5 = 0

    test_len = 0
    perplexity_loss = 0.0

    net.eval()
    with torch.no_grad():
        for data in tqdm(testloader):
            try:
                data, target = mask_tokens(
                    data, tokenizer, mlm_probability=0.15, device=device
                )
                data, target = Variable(data).to(device=device), Variable(target).to(
                    device=device
                )

                outputs = net(data, labels=target)

                loss: torch.Tensor = outputs[0]
                test_loss += loss.item()
                perplexity_loss += loss.item()

                acc = accuracy(
                    outputs[1].reshape(-1, outputs[1].shape[2]),
                    target.reshape(-1),
                    topk=(1, 5),
                )

                correct += acc[0].item()

            except Exception as ex:
                print(f"Testing failed as {ex}")
                break
            test_len += len(target)

        test_len = max(test_len, 1)
        # loss function averages over batch size
        test_loss /= len(testloader)
        perplexity_loss /= len(testloader)

        sum_loss = test_loss * test_len

        # in NLP, we care about the perplexity of the model
        acc = round(correct / test_len, 4)
        acc_5 = round(top_5 / test_len, 4)
        test_loss = round(test_loss, 4)

        testRes = {
            "acc": acc,
            "acc_5": acc_5,
            "top_1": correct,
            "top_5": top_5,
            "perplexity_loss": perplexity_loss,
            "sum_test_loss": sum_loss,
        }

    return test_loss, test_len, testRes


def google_speech_testing_loop(
    testloader: DataLoader,
    device: torch.device,
    net: Module,
    criterion: Module,
    **kwargs,
):
    test_loss = 0
    acc_top_1 = 0
    acc_top_5 = 0
    test_len = 0

    net.eval()
    with torch.no_grad():
        for data, target in tqdm(testloader):
            try:
                data, target = Variable(data).to(device=device), Variable(target).to(
                    device=device
                )
                data = torch.unsqueeze(data, 1)

                output = net(data)

                loss: torch.Tensor = criterion(output, target)
                test_loss += loss.item()
                acc = accuracy(output, target, topk=(1, 5))

                acc_top_1 += acc[0].item()
                acc_top_5 += acc[1].item()

            except Exception as ex:
                print(f"Testing failed as {ex}")
                break
            test_len += len(target)

        # Number of test samples
        test_len = max(test_len, 1)
        # Test loss averages over number of batches
        test_loss /= len(testloader)
        # Accuracy averages over number of batches
        acc_top_1 = round(acc_top_1 / test_len, 4)
        acc_top_5 = round(acc_top_5 / test_len, 4)
        test_loss = round(test_loss, 4)

        test_metrics = {
            "acc_top_1": acc_top_1,
            "acc_top_5": acc_top_5,
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
    acc_top_1 = 0
    acc_top_5 = 0
    test_len = 0

    net.eval()
    with torch.no_grad():
        for data, target in tqdm(testloader):
            try:
                data, target = Variable(data).to(device=device), Variable(target).to(
                    device=device
                )

                output = net(data)

                loss: torch.Tensor = criterion(output, target)
                test_loss += loss.item()
                acc = accuracy(output, target, topk=(1, 5))

                acc_top_1 += acc[0].item()
                acc_top_5 += acc[1].item()

            except Exception as ex:
                print(f"Testing failed as {ex}")
                break
            test_len += len(target)

        # Number of test samples
        test_len = max(test_len, 1)
        # Test loss averages over number of batches
        test_loss /= len(testloader)
        # Accuracy averages over number of batches
        acc_top_1 = round(acc_top_1 / test_len, 4)
        acc_top_5 = round(acc_top_5 / test_len, 4)
        test_loss = round(test_loss, 4)

        test_metrics = {
            "acc_top_1": acc_top_1,
            "acc_top_5": acc_top_5,
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
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python models/testing_loops.py output_dir="/nfs-share/ls985/pollen_worker/outputs/2023-09-01/16-20-25" task="reddit"
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python models/testing_loops.py output_dir="/nfs-share/ls985/pollen_worker/outputs/2023-09-01/16-20-16" task="google_speech"
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python models/testing_loops.py output_dir="/nfs-share/ls985/pollen_worker/outputs/2023-09-01/16-20-03" task="openimage"
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python models/testing_loops.py output_dir="/nfs-share/ls985/pollen_worker/outputs/2023-09-01/16-19-57" task="shakespeare_memory"
    import pickle
    from collections import OrderedDict
    from logging import INFO
    from multiprocessing import Pool
    from pathlib import Path

    import psutil
    from flwr.common import parameters_to_ndarrays
    from flwr.common.logger import log
    from flwr.common.typing import NDArrays
    from torch.utils.data import ConcatDataset

    from pollen_utils import get_clients_population_dict, get_model

    def set_parameters(parameters: NDArrays, net: Module = None, device="cuda"):
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
    client_ids = list(cid_samples_dict.keys())
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
    batch_sizes = {
        "reddit": 20,
        "google_speech": 512,
        "openimage": 1024,
        "shakespeare_memory": 256,
    }
    testloader = DataLoader(
        testset, batch_size=batch_sizes[cfg.task.name], shuffle=False, pin_memory=True
    )
    # Create results .csv file
    results_file = root_dir / "offline_eval_results.csv"
    # Get the models' performance
    for i, parameters_file in enumerate(root_dir.glob("parameters_aggregated_*")):
        round = int(parameters_file.name.split("_")[-1])
        with open(parameters_file, "rb") as f:
            parameters: NDArrays = pickle.load(f)
        net = set_parameters(parameters=parameters_to_ndarrays(parameters))
        net.to(device="cuda")
        net.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="mean").to(device="cuda")
        test_res = test_loop(
            testloader=testloader,
            device="cuda",
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
        with open(results_file, "a") as f:
            metrics = ",".join([f"{v}" for _, v in test_res[2].items()])
            f.write(f"{round},{test_res[0]},{metrics}\n")
        log(INFO, f"Round {round}, test loss: {test_res[0]}, metrics: {metrics}")


if __name__ == "__main__":
    main()
