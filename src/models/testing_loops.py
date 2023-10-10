from typing import List, Tuple

import hydra
import torch
import transformers
import wandb
import yaml
from omegaconf import DictConfig
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AlbertTokenizer
from transformers.modeling_outputs import MaskedLMOutput

from datasets.nlp_util import mask_tokens
from utils import wandb_init

transformers.logging.set_verbosity_error()


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
    """Computes the accuracy over the k top predictions for the specified values of
    k.
    """
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


@hydra.main(config_path="../conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    import pickle
    import time
    from logging import INFO
    from pathlib import Path

    import psutil
    from flwr.common import parameters_to_ndarrays
    from flwr.common.logger import log
    from flwr.common.typing import Parameters

    from datasets.nlp_util import get_collate_fn
    from pollen_utils import get_centralised_eval_set, get_device, get_model
    from utils import set_parameters

    device = get_device()

    log(
        INFO,
        "Offline evaluation of task %s. Using output_dir: %s",
        cfg.task.name,
        cfg.output_dir,
    )
    s_t = time.time()
    # Set the root directory
    root_dir = Path(cfg.output_dir)
    # Get test_loop fn
    test_loop = get_testing_loop(name=cfg.task.name)
    # Get number of available cpu cores
    try:
        n_cpus = len(psutil.Process().cpu_affinity())
    except AttributeError:
        n_cpus = psutil.cpu_count()
    # Get the test set
    # NOTE: The `n_clients` parameter, when greater than one, can limit the clients
    # to be used for the evaluation to the biggest `n_clients`
    testset, tokenizer = get_centralised_eval_set(
        name=cfg.task.name, n_clients=cfg.task.n_clients, seed=cfg.seed
    )
    log(INFO, f"Test set size: {len(testset)}")
    # Instantiate the test loader
    batch_sizes = {
        "reddit": 375,  # Fills up the VRAM
        "google_speech": 1024,  # Doesn't really matter: too few sample
        "openimage": 1200,  # If increased, it crashes
        "shakespeare_memory": 256,  # Doesn't really matter: too few sample
    }
    testloader = DataLoader(
        testset,
        batch_size=batch_sizes[cfg.task.name],
        shuffle=False,
        pin_memory=True,
        collate_fn=get_collate_fn(tokenizer=tokenizer)
        if tokenizer is not None
        else None,
        num_workers=n_cpus,
    )
    with open(root_dir / ".hydra" / "config.yaml", "r") as f:
        wandb_config = yaml.safe_load(f)

    log(INFO, f"Time to get the eval dataloader: {time.time() - s_t}")
    # Create results .csv file
    results_file = root_dir / "offline_eval_results.csv"
    net = None
    # Get the models' performance
    with wandb_init(
        cfg.use_wandb,
        **cfg.wandb.setup,
        settings=wandb.Settings(start_method="thread"),
        config=wandb_config,  # type: ignore
    ):
        torch.backends.cudnn.benchmark = True
        for i, parameters_file in enumerate(root_dir.glob("parameters_aggregated_*")):
            round = int(parameters_file.name.split("_")[-1])
            with open(parameters_file, "rb") as f:
                parameters = pickle.load(f)
            if isinstance(parameters, Parameters):
                parameters = parameters_to_ndarrays(parameters)
            net = get_model(name=cfg.task.name)
            set_parameters(parameters=parameters, net=net, device=device)
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
                    "A file containing the round number,"
                    " the average test loss, and metrics (%s) will be written",
                    metrics,
                )
            with open(results_file, "a") as f:
                metrics = ",".join([f"{v}" for _, v in test_res[2].items()])
                f.write(f"{round},{test_res[0]},{metrics}\n")

            log(INFO, f"Round {round}, test loss: {test_res[0]}, metrics: {metrics}")
            # break
            if cfg.use_wandb:
                wandb.log({"test_loss": test_res[0], **test_res[2]}, step=round)


if __name__ == "__main__":
    main()
