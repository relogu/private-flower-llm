"""Definition for the state of a Minio instance."""
from typing import Callable

from minio import Minio


class MinioState(object):
    """Define the state of a Minio instance."""

    def __init__(
        self,
        client: Minio,
        run_uuid: str,
        endpoint_id: str,  # node_manager_uuid or server id
        bucket_name: str,
        file_size: int = 1024 * 1024 * 100,  # 100MB
        throw_on_error: bool = True,
        timeout_in_seconds: int = 60 * 60,  # 1 hour
        log: Callable | None = None,
    ) -> None:
        if not isinstance(client, Minio):
            raise TypeError("client is not an instance of Minio")
        if not isinstance(run_uuid, str):
            raise TypeError("run_uuid is not a string")
        if not isinstance(endpoint_id, str):
            raise TypeError("node_manager_uuid is not a string")
        if not isinstance(bucket_name, str) or len(bucket_name) < 1:
            raise TypeError("bucket_name is not a valid string")
        if not isinstance(file_size, int) or file_size < 1:
            raise TypeError("file_size is not a positive non-zero integer")
        if not isinstance(throw_on_error, bool):
            raise TypeError("throw_on_error is not a Boolean value")
        if not isinstance(timeout_in_seconds, int) or timeout_in_seconds < 1:
            raise TypeError("timeout_in_seconds is not a positive non-zero integer")
        if not (callable(log) or log is None):
            raise TypeError("log_method is not a positive non-zero integer")

        self.client: Minio = client
        self.run_uuid: str = run_uuid
        self.endpoint_id: str = endpoint_id
        self.bucket_name: str = bucket_name
        self.file_size: int = file_size
        self.throw_on_error: bool = True
        self.timeout_in_seconds: int = timeout_in_seconds
        self.log: Callable | None = log
