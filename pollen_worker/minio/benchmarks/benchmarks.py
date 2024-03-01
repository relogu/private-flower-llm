"""Module for the benchmarks."""
import configparser
import copy
import csv
import gc
import logging
import sys
import time
import uuid
from pathlib import Path

import hydra
import numpy as np
from flwr.common import NDArrays, log, ndarrays_to_parameters
from minio import Minio
from omegaconf import DictConfig, OmegaConf

from pollen_worker.clients.llm_client_functions import get_raw_model_parameters
from pollen_worker.minio.minio_state import MinioState
from pollen_worker.minio.minio_tools import (
    _get_justified_number,
    pull_parameters,
    push_parameters,
)


class Benchmarks:
    """Class for the benchmarks."""

    def __init__(self, parameters_as_ndarrays: NDArrays) -> None:
        self.parameters_as_ndarrays = parameters_as_ndarrays
        self.parameters_as_tensors = ndarrays_to_parameters(parameters_as_ndarrays)
        home = Path.expanduser(Path("~"))
        config_file_path = home / ".aws" / "credentials"
        if not Path.is_file(config_file_path):
            raise ValueError("Invalid config_file_path")
        config = configparser.ConfigParser()
        config.read(config_file_path)
        access_key_id = config["default"]["aws_access_key_id"]
        aws_secret_access_key = config["default"]["aws_secret_access_key"]
        if not (isinstance(access_key_id, str) and len(access_key_id) > 0):
            raise TypeError("Invalid access_key_id")
        if not (
            isinstance(aws_secret_access_key, str) and len(aws_secret_access_key) > 0
        ):
            raise TypeError("Invalid aws_secret_access_key")
        self.client = Minio(
            "mauao.cl.cam.ac.uk:9000",
            access_key=access_key_id,
            secret_key=aws_secret_access_key,
            secure=False,
        )

        self.bucket_name = "test"
        if not self.client.bucket_exists(self.bucket_name):
            raise ValueError("Invalid aws_secret_access_key")

        self.run_uuid = str(uuid.uuid4())
        self.node_manager_uuid = str(uuid.uuid4())

        logger = logging.getLogger("MinIO_logger")
        logger.addHandler(logging.StreamHandler(sys.stdout))
        self.log = logger.log

    @staticmethod
    def _get_stylizes_size(size: int) -> str:
        if size >= 1024 * 1024:
            return f"{size / (1024 * 1024):,} MB"
        else:
            return f"{size / 1024} kB"

    def minimum_file_size_vs_speed(self) -> None:
        """Execute the minimum file size vs speed benchmark."""
        kb = 1024
        mb = 1024 * kb
        file_sizes = []
        file_size = 32 * mb
        while file_size <= 512 * mb:
            file_sizes.append(file_size)
            file_size *= 2

        log(
            logging.INFO,
            (
                "\nBenchmark file sizes (kB):"
                f" {[int(value / kb) for value in file_sizes]}\n"
            ),
        )

        results: list[list[float]] = []

        results_index = 0

        server_round = 1

        for file_size in file_sizes:
            results.append([])
            results[results_index] = []
            results[results_index].append(file_size)

            size_str = _get_justified_number(file_size, 36)

            state = MinioState(
                self.client,
                size_str,
                self.node_manager_uuid,
                self.bucket_name,
                file_size,
                True,
                60 * 30,
                self.log,
            )

            for results_index, _attempt in enumerate(range(1, 2, 1)):
                gc.collect()
                time.sleep(1)

                start_time = time.time()
                push_parameters(state, server_round, self.parameters_as_ndarrays)
                pulled_parameters = pull_parameters(state, server_round)
                if not isinstance(pulled_parameters, list):
                    raise ConnectionError("Failed to pull parameters")
                end_time = time.time()

                time_diff = end_time - start_time

                log(
                    logging.INFO,
                    (
                        f"File size: {Benchmarks._get_stylizes_size(file_size)};"
                        f" Time: {time_diff} seconds"
                    ),
                )

                # Check the integrity of the pulled parameters
                if len(self.parameters_as_ndarrays) != len(pulled_parameters):
                    raise ValueError("Invalid parameters length")
                for index in range(0, len(self.parameters_as_ndarrays), 1):
                    if not np.array_equal(
                        self.parameters_as_ndarrays[index], pulled_parameters[index]
                    ):
                        raise ValueError("Parameter arrays are not equal")

                results[results_index].append(time_diff)

        home = Path.expanduser(Path("~"))
        csv_file_path = home / "benchmarks" / "mpt-1b.csv"
        with open(csv_file_path, "w", encoding="locale") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(results)

        log(logging.INFO, "✅ All done!")


@hydra.main(config_path="../../conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the benchmarks."""
    _llm_config = cfg.llm_config
    OmegaConf.resolve(_llm_config)
    OmegaConf.set_struct(_llm_config, False)
    model_parameters = get_raw_model_parameters(copy.deepcopy(_llm_config))

    benchmarks = Benchmarks(model_parameters)
    benchmarks.minimum_file_size_vs_speed()


if __name__ == "__main__":
    main()
