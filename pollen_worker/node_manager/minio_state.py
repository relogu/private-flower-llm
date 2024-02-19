from minio import Minio

class MinioState(object):

    def __init__(
        self,
        client: Minio,
        bucket_name: str,
        server_round: int,
        buffer_length: int,
        run_uuid: str,
        node_manager_uuid: str
    ) -> None:

        if not isinstance(client, Minio):
            raise TypeError("client is not an instance of Minio")
        if not isinstance(bucket_name, str) or len(bucket_name) < 1:
            raise TypeError("bucket_name is not a valid string")
        if not isinstance(server_round, int) or server_round < 0:
            raise TypeError("server_round is not a positive integer")
        if not isinstance(buffer_length, int) or buffer_length < 1:
            raise TypeError("buffer_length is not a positive non-zero integer")
        if not isinstance(run_uuid, str) or len(run_uuid) != 36:
            raise TypeError("run_uuid is not a valid UUID")
        if not isinstance(node_manager_uuid, str) or len(node_manager_uuid) != 36:
            raise TypeError("node_manager_uuid is not a valid UUID")

        self.client: Minio = client
        self.bucket_name: str = bucket_name
        self.server_round: int = server_round
        self.buffer_length: int = buffer_length
        self.run_uuid: str = run_uuid
        self.node_manager_uuid: str = node_manager_uuid
        #TODO: minimum bucket file size
