"""Module for testing the MinIO tools."""
import configparser
import logging
import os
import random
import sys
import unittest
import uuid

import numpy as np
from flwr.common import NDArrays
from flwr.server.history import History
from minio import Minio

from pollen_worker.minio.minio_state import MinioState
from pollen_worker.minio.minio_tools import (
    SERVER_GLOBAL_MODEL_FOLDER,
    _get_params_folder_path,
    pull_parameters,
    pull_server_state,
    push_parameters,
    push_server_state,
)
from pollen_worker.server_state import ServerState


class TestMinio(unittest.TestCase):
    """Class for unit test of the Minio."""

    def setUp(self):
        """Set up the Minio client."""
        home = os.path.expanduser("~")
        config_file_path = os.path.join(home, ".aws", "credentials")
        self.assertTrue(os.path.isfile(config_file_path))
        config = configparser.ConfigParser()
        config.read(config_file_path)
        access_key_id = config["default"]["aws_access_key_id"]
        aws_secret_access_key = config["default"]["aws_secret_access_key"]
        self.assertTrue(isinstance(access_key_id, str) and len(access_key_id) > 0)
        self.assertTrue(
            isinstance(aws_secret_access_key, str) and len(aws_secret_access_key) > 0
        )
        self.client = Minio(
            "mauao.cl.cam.ac.uk:9000",
            access_key=access_key_id,
            secret_key=aws_secret_access_key,
            secure=False,
        )

        self.bucket_name = "test"
        self.assertTrue(self.client.bucket_exists(self.bucket_name))

        self.run_uuid = str(uuid.uuid4())
        self.node_manager_uuid = str(uuid.uuid4())

        self.mock_parameters = TestMinio.get_mock_parameters()

        logger = logging.getLogger("MinIO_logger")
        logger.addHandler(logging.StreamHandler(sys.stdout))
        self.log = logger.log

    @staticmethod
    def get_mock_parameters() -> NDArrays:
        """Return the mock parameters."""
        mock_parameters: NDArrays = []
        for _index in range(0, random.randint(100, 1000), 1):
            list_len = random.randint(1, 1000)
            ordered_list = list(range(0, list_len, 1))
            randomized_list = random.sample(ordered_list, list_len)
            mock_parameters.append(np.array(randomized_list))
        return mock_parameters

    def test_push_and_pull_parameters(self):
        """Test the push and pull of the parameters."""
        round = 1

        # Push
        state = MinioState(
            self.client,
            self.run_uuid,
            self.node_manager_uuid,
            self.bucket_name,
            1024 * 100,
            True,
            30,
            self.log,
        )
        self.assertTrue(push_parameters(state, round, self.mock_parameters))

        # Pull
        pulled_parameters = pull_parameters(state, round)
        self.assertTrue(isinstance(pulled_parameters, list))

        # Check the integrity of the pulled parameters
        self.assertEqual(len(self.mock_parameters), len(pulled_parameters))
        for index in range(0, len(self.mock_parameters), 1):
            self.assertTrue(
                np.array_equal(self.mock_parameters[index], pulled_parameters[index])
            )

    def test_server_state(self):
        """Test the server state."""
        global_model_nda = self.mock_parameters
        momentum_nda = TestMinio.get_mock_parameters()

        minio_state = MinioState(
            self.client,
            self.run_uuid,
            "server",
            self.bucket_name,
            1024 * 100,
            True,
            30,
            self.log,
        )

        mock_history = History()
        mock_history.losses_distributed.append((12, 3.4))

        server_state = ServerState(
            "server",  # id
            1,  # round
            momentum_nda,  # momentum
            12.34,  # elapsed_time_in_seconds
            mock_history,  # history
        )

        self.assertTrue(push_server_state(minio_state, server_state))

        global_model_pth = f"{_get_params_folder_path(minio_state, server_state.round)}"
        f"/{SERVER_GLOBAL_MODEL_FOLDER}"
        self.assertTrue(
            push_parameters(
                minio_state, server_state.round, global_model_nda, global_model_pth
            )
        )

        retrieved_state = pull_server_state(minio_state, server_state.round)

        self.assertTrue(isinstance(retrieved_state, ServerState))

        self.assertEqual(server_state.id, retrieved_state.id)
        self.assertEqual(server_state.round, retrieved_state.round)
        self.assertEqual(
            server_state.elapsed_time_in_seconds,
            retrieved_state.elapsed_time_in_seconds,
        )
        self.assertEqual(retrieved_state.history.losses_distributed[0], (12, 3.4))

        self.assertEqual(len(global_model_nda), len(retrieved_state.global_model))
        for index in range(0, len(global_model_nda), 1):
            self.assertTrue(
                np.array_equal(
                    global_model_nda[index], retrieved_state.global_model[index]
                )
            )

        self.assertEqual(len(momentum_nda), len(retrieved_state.momentum))
        for index in range(0, len(momentum_nda), 1):
            self.assertTrue(
                np.array_equal(momentum_nda[index], retrieved_state.momentum[index])
            )


if __name__ == "__main__":
    unittest.main()
