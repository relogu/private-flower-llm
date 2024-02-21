from minio import Minio

class MinioState(object):

    def __init__(
        self,
        client: Minio,
        run_uuid: str,
        node_manager_uuid: str,
        server_round: int,
        bucket_name: str,
        minimum_file_size: int = 1024 * 1024 * 100 # 100MB
    ) -> None:

        if not isinstance(client, Minio):
            raise TypeError("client is not an instance of Minio")
        if not isinstance(run_uuid, str) or len(run_uuid) != 36:
            raise TypeError("run_uuid is not a valid UUID")
        if not isinstance(node_manager_uuid, str) or len(node_manager_uuid) != 36:
            raise TypeError("node_manager_uuid is not a valid UUID")
        if not isinstance(server_round, int) or server_round < 0:
            raise TypeError("server_round is not a positive integer")
        if not isinstance(bucket_name, str) or len(bucket_name) < 1:
            raise TypeError("bucket_name is not a valid string")
        if not isinstance(minimum_file_size, int) or minimum_file_size < 1:
            raise TypeError("minimum_file_size is not a positive non-zero integer")

        self.client: Minio = client
        self.run_uuid: str = run_uuid
        self.node_manager_uuid: str = node_manager_uuid
        self.server_round: int = server_round
        self.bucket_name: str = bucket_name
        self.minimum_file_size: int = minimum_file_size
