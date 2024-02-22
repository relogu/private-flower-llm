import random
import time
import unittest
import os
import configparser
import uuid
from minio import Minio
from flwr.common import NDArrays, ndarrays_to_parameters, parameters_to_ndarrays, Parameters
import numpy as np
from pollen_worker.node_manager.minio_state import MinioState
from pollen_worker.node_manager.minio_tools import MinioTools
from minio import Minio

class TestMinioTools(unittest.TestCase):

    def setUp(self):
        home = os.path.expanduser("~")
        config_file_path = os.path.join(home, ".aws", "credentials")
        self.assertTrue(os.path.isfile(config_file_path))
        config = configparser.ConfigParser()
        config.read(config_file_path)
        access_key_id = config["default"]["aws_access_key_id"]
        aws_secret_access_key = config["default"]["aws_secret_access_key"]
        self.assertTrue(isinstance(access_key_id, str) and len(access_key_id) > 0)
        self.assertTrue(isinstance(aws_secret_access_key, str) and len(aws_secret_access_key) > 0)
        self.client = Minio("mauao.cl.cam.ac.uk:9000",
            access_key = access_key_id,
            secret_key = aws_secret_access_key,
            secure = False
        )

        self.bucket_name = "test"
        self.assertTrue(self.client.bucket_exists(self.bucket_name))

        self.run_uuid = str(uuid.uuid4())
        self.node_manager_uuid = str(uuid.uuid4())

        self.mock_parameters: NDArrays = []
        for index in range(0, random.randint(100, 1000), 1):
             list_len = random.randint(1, 1000)
             ordered_list = list(range(0, list_len, 1))
             randomized_list = random.sample(ordered_list, list_len)
             self.mock_parameters.append(np.array(randomized_list))

    def test_push_and_pull(self):

        # Push
        state = MinioState(self.client, self.run_uuid, self.node_manager_uuid, 1, self.bucket_name, 1024 * 100)
        self.assertTrue(MinioTools.push_parameters(state, self.mock_parameters))

        # Pull
        pulled_parameters = MinioTools.pull_parameters(state)
        self.assertTrue(isinstance(pulled_parameters, list))

        # Check the integrity of the pulled parameters
        self.assertEqual(len(self.mock_parameters), len(pulled_parameters))
        for index in range(0, len(self.mock_parameters), 1):
            self.assertTrue(np.array_equal(self.mock_parameters[index], pulled_parameters[index]))

if __name__ == '__main__':
    unittest.main()
