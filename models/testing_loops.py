import torch
from torch.autograd import Variable
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer

from datasets.nlp_util import mask_tokens


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

    for data, target in testloader:
        try:
            data, target = mask_tokens(
                data, tokenizer, mlm_probability=0.15, evice=device
            )
            data, target = Variable(data).to(device=device), Variable(target).to(
                device=device
            )

            outputs = net(data, labels=target)

            loss = outputs[0]
            test_loss += loss.data.item()
            perplexity_loss += loss.data.item()

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
    correct = 0
    top_5 = 0

    test_len = 0

    for data, target in testloader:
        try:
            data, target = Variable(data).to(device=device), Variable(target).to(
                device=device
            )
            data = torch.unsqueeze(data, 1)

            output = net(data)
            loss = criterion(output, target)

            test_loss += loss.data.item()  # Variable.data
            acc = accuracy(output, target, topk=(1, 5))

            correct += acc[0].item()
            top_5 += acc[1].item()

        except Exception as ex:
            print(f"Testing failed as {ex}")
            break
        test_len += len(target)

    test_len = max(test_len, 1)
    # loss function averages over batch size
    test_loss /= len(testloader)

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
        "sum_test_loss": sum_loss,
    }

    return test_loss, test_len, testRes


def general_testing_loop(
    testloader: DataLoader,
    device: torch.device,
    net: Module,
    criterion: Module,
    **kwargs,
):
    test_loss = 0
    correct = 0
    top_5 = 0

    test_len = 0

    for data, target in testloader:
        try:
            data, target = Variable(data).to(device=device), Variable(target).to(
                device=device
            )

            output = net(data)

            loss = criterion(output, target)
            test_loss += loss.data.item()
            acc = accuracy(output, target, topk=(1, 5))

            correct += acc[0].item()
            top_5 += acc[1].item()

        except Exception as ex:
            print(f"Testing failed as {ex}")
            break
        test_len += len(target)

    test_len = max(test_len, 1)
    # loss function averages over batch size
    test_loss /= len(testloader)

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
        "sum_test_loss": sum_loss,
    }

    return test_loss, test_len, testRes


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        return res
