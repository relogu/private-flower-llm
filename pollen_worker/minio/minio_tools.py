"""Tools to interact with MinIO."""

import binascii
import gc
import hashlib
import io
import json
import pickle
import time
from logging import ERROR, INFO

from flwr.common import (
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.history import History
from minio import S3Error
from minio.helpers import ObjectWriteResult

from pollen_worker.minio.minio_state import MinioState
from pollen_worker.server_state import ServerState, ServerStateWithGlobalModel

METADATA_FILE_NAME = "metadata.json"

SERVER_GLOBAL_MODEL_FOLDER = "global_model"
SERVER_MOMENTUM_FOLDER = "momentum"
SERVER_HISTORY_FILE = "history.pkl"
SERVER_STATE_FILE = "state.json"


def _get_full_file_path(state: MinioState, server_round: int, file_name: str) -> str:
    return f"{_get_params_folder_path(state, server_round)}/{file_name}"


def _get_params_folder_path(state: MinioState, server_round: int) -> str:
    return f"{state.run_uuid}/{_get_justified_number(server_round)}/{state.endpoint_id}"


def _get_justified_number(number: int, digits: int = 8) -> str:
    return str(number).rjust(digits, "0")


def _pull_single_parameters_file(
    state: MinioState,
    server_round: int,
    file_list: list,
    file_index: int,
    minio_folder_path: str | None = None,
) -> bytes | None:
    """Pull a single file of model parameters from MinIO."""
    file_name: str
    file_hash: str
    try:
        file_name = file_list[file_index]["name"]
        file_hash = file_list[file_index]["sha3_256"]
    except Exception:
        _value_error(state, "Failed to retrieve file_name of file_hash from the list")
        return None
    if len(file_name) < 15:  # noqa: PLR2004
        _value_error(state, "The metadata contain an invalid file name")
        return None
    if len(file_hash) != 64:  # noqa: PLR2004
        _value_error(state, "The metadata contain an invalid hash code")
        return None

    full_file_path: str
    if minio_folder_path is None:
        full_file_path = _get_full_file_path(state, server_round, file_name)
    else:
        full_file_path = f"{minio_folder_path}/{file_name}"

    file_bytes = _pull_single_file(state, full_file_path)
    if not isinstance(file_bytes, bytes):
        return None
    real_file_hash = hashlib.sha3_256()
    real_file_hash.update(file_bytes)
    actual_file_hash = binascii.hexlify(real_file_hash.digest()).decode("utf-8")
    if real_file_hash != actual_file_hash:
        _value_error(
            state,
            (
                f"The computed hash code of file '{full_file_path}' differs from the"
                " one stored in metadata.json"
            ),
        )
        return None
    return file_bytes


def _pull_single_file(state: MinioState, full_file_path: str) -> bytes | None:
    """Pull a single file from MinIO."""
    start_time = time.time()
    elapsed_time = 0.0
    try_again = True
    pull_successful = False
    number_of_attemts = 0
    file_bytes: bytes
    while try_again:
        number_of_attemts += 1
        try:
            response = state.client.get_object(state.bucket_name, full_file_path)
            file_bytes = response.read()
            response.close()
            response.release_conn()
            pull_successful = True
        except Exception as ex:
            if isinstance(ex, S3Error) and ex.code == "NoSuchKey":
                message = "🚨 The requested file does not exist on MinIO:"
                f" {full_file_path}"
                _log_error(state, message)
                raise ValueError(message) from ex
            _log_error(state, f"Failed to pull the file: {full_file_path}")
        elapsed_time = time.time() - start_time
        try_again = (not pull_successful) and (elapsed_time < state.timeout_in_seconds)
        if try_again:
            _log_info(
                state,
                (
                    f"Attempt #{number_of_attemts}; trying again to pull the file from"
                    f" MinIO: {full_file_path}"
                ),
            )
            gc.collect()
            time.sleep(3)
    if pull_successful and number_of_attemts > 1:
        _log_info(
            state,
            (
                f"Successfully pulled the file from MinIO after {number_of_attemts}"
                f" attempts: {full_file_path}"
            ),
        )
    if not pull_successful:
        _connection_error(
            state,
            (
                f"🚨 Timed out after {number_of_attemts} attempt(s) and"
                f" {int(elapsed_time)} second(s). Completely failed to pull the file"
                f" from MinIO: {full_file_path}"
            ),
        )
        return None
    return file_bytes


def pull_parameters(
    state: MinioState, server_round: int, minio_folder_path: str | None = None
) -> NDArrays:
    """Pull model parameters from MinIO."""
    tensor_type = ""
    tensor_sizes = []
    file_list = []

    metadata_file_path: str
    if minio_folder_path is None:
        metadata_file_path = _get_full_file_path(
            state, server_round, METADATA_FILE_NAME
        )
    else:
        metadata_file_path = f"{minio_folder_path}/{METADATA_FILE_NAME}"

    metadata_bytes = _pull_single_file(state, metadata_file_path)
    if not isinstance(metadata_bytes, bytes):
        message = "metadata_bytes are not of type bytes"
        _log_error(state, message)
        raise TypeError(state, message)
    try:
        metadata_json_string = metadata_bytes.decode("utf-8")
        json_root = json.loads(metadata_json_string)
        tensor_type = json_root["tensorType"]
        tensor_sizes = json_root["tensorSizes"]
        file_list = json_root["files"]
    except Exception as ex:
        _type_error(
            state,
            f"Failed to deserialize the metadata JSON file: {metadata_file_path}",
            ex,
        )
    if len(tensor_sizes) == 0 or len(file_list) == 0:
        _value_error(state, "Failed to retrieve values from the metadata JSON file")

    tensors: list[bytes] = []
    current_file_index = 0
    current_file_content: bytes | None
    current_file_pointer = 0
    current_tensor_index = 0
    current_tensor_content = bytearray()
    all_files_processed = False
    all_done = False

    current_file_content = _pull_single_parameters_file(
        state, server_round, file_list, current_file_index, minio_folder_path
    )
    if not isinstance(current_file_content, bytes):
        message = "current_file_content are not of type bytes"
        _log_error(state, message)
        raise TypeError(state, message)
    total_size_of_files = len(current_file_content)
    total_size_of_tensors = 0

    while not all_done:
        available_space = tensor_sizes[current_tensor_index] - len(
            current_tensor_content
        )
        if available_space == 0 or all_files_processed:
            tensors.append(bytes(current_tensor_content))
            current_tensor_content.clear()
            total_size_of_tensors += tensor_sizes[current_tensor_index]
            if all_files_processed:
                all_done = True
            elif current_tensor_index < len(tensor_sizes) - 1:
                current_tensor_index += 1
        elif available_space > 0:
            current_file_size = len(current_file_content) - current_file_pointer
            if available_space >= current_file_size:
                current_tensor_content.extend(
                    current_file_content[
                        current_file_pointer : len(current_file_content)
                    ]
                )
                current_file_index += 1
                current_file_pointer = 0
                if current_file_index < len(file_list):
                    current_file_content = _pull_single_parameters_file(
                        state,
                        server_round,
                        file_list,
                        current_file_index,
                        minio_folder_path,
                    )
                    if not isinstance(current_file_content, bytes):
                        message = "current_file_content is not of type bytes"
                        _log_error(state, message)
                        raise TypeError(state, message)
                    total_size_of_files += len(current_file_content)
            else:
                current_tensor_content.extend(
                    current_file_content[
                        current_file_pointer : current_file_pointer + available_space
                    ]
                )
                current_file_pointer += available_space
            all_files_processed = len(file_list) <= current_file_index
        else:
            message = "available_space cannot be a negative number"
            _log_error(state, message)
            raise ValueError(state, message)

    if total_size_of_tensors != total_size_of_files:
        message = "Tensor and file sizes are not the same"
        _log_error(state, message)
        raise ValueError(state, message)

    return parameters_to_ndarrays(Parameters(tensors, tensor_type))


def _push_single_file(
    state: MinioState, full_file_path: str, file_content: bytes | bytearray
) -> bool:
    """Pull a single file from MinIO."""
    start_time = time.time()
    elapsed_time = 0.0
    try_again = True
    push_successful = False
    number_of_attemts = 0
    while try_again:
        number_of_attemts += 1
        try:
            write_result = state.client.put_object(
                state.bucket_name,
                full_file_path,
                io.BytesIO(file_content),
                length=len(file_content),
            )
            if not (
                isinstance(write_result, ObjectWriteResult)
                and len(write_result.object_name) > 0
            ):
                _connection_error(
                    state, f"Failed to push the file to MinIO: {full_file_path}"
                )
            push_successful = True
        except Exception as ex:
            _log_error(state, f"Failed to push the file to MinIO: {full_file_path}", ex)
        elapsed_time = time.time() - start_time
        try_again = (not push_successful) and (elapsed_time < state.timeout_in_seconds)
        if try_again:
            _log_info(
                state,
                (
                    f"Attempt #{number_of_attemts}; trying again to push the file to"
                    f" MinIO: {full_file_path}"
                ),
            )
            gc.collect()
            time.sleep(3)
    if push_successful and number_of_attemts > 1:
        _log_info(
            state,
            (
                f"Successfully pushed the file to MinIO after {number_of_attemts}"
                f" attempts: {full_file_path}"
            ),
        )
    if not push_successful:
        _connection_error(
            state,
            (
                f"🚨 Timed out after {number_of_attemts} attempt(s) and"
                f" {int(elapsed_time)} second(s). Completely failed to push the file to"
                f" MinIO: {full_file_path}"
            ),
        )
    return push_successful


def push_parameters(
    state: MinioState,
    server_round: int,
    parameters: NDArrays | Parameters,
    minio_folder_path: str | None = None,
) -> bool:
    """Push model parameters to MinIO."""
    if isinstance(parameters, list):
        parameters = ndarrays_to_parameters(parameters)
    elif not isinstance(parameters, Parameters):
        _type_error(
            state,
            "parameters are not an instance of NDArrays or Parameters",
        )

    tensors = parameters.tensors
    file_list = []
    current_file_id = 1
    tensor_sizes = []
    current_file_content = bytearray()
    current_tensor_index = 0
    current_tensor_pointer = 0
    all_tensors_processed = False
    all_done = False

    total_size_of_files = 0
    total_size_of_tensors = 0

    while not all_done:
        available_space = state.file_size - len(current_file_content)
        if available_space == 0 or all_tensors_processed:
            current_file_name = f"{_get_justified_number(current_file_id)}.params"
            file_hash = hashlib.sha3_256()
            file_hash.update(current_file_content)
            current_file_hash = binascii.hexlify(file_hash.digest()).decode("utf-8")
            file_list.append({"name": current_file_name, "sha3_256": current_file_hash})
            full_file_path: str
            if minio_folder_path is None:
                full_file_path = _get_full_file_path(
                    state, server_round, current_file_name
                )
            else:
                full_file_path = f"{minio_folder_path}/{current_file_name}"

            result = _push_single_file(state, full_file_path, current_file_content)
            if not result:
                return False

            if all_tensors_processed:
                all_done = True
            else:
                current_file_id += 1
            total_size_of_files += len(current_file_content)
            current_file_content.clear()
        elif available_space > 0:
            current_tensor_size = (
                len(tensors[current_tensor_index]) - current_tensor_pointer
            )
            if available_space >= current_tensor_size:
                current_file_content.extend(
                    tensors[current_tensor_index][
                        current_tensor_pointer : len(tensors[current_tensor_index])
                    ]
                )
                tensor_sizes.append(len(tensors[current_tensor_index]))
                total_size_of_tensors += len(tensors[current_tensor_index])
                current_tensor_index += 1
                current_tensor_pointer = 0
            else:
                current_file_content.extend(
                    tensors[current_tensor_index][
                        current_tensor_pointer : current_tensor_pointer
                        + available_space
                    ]
                )
                current_tensor_pointer += available_space
            all_tensors_processed = len(tensors) == current_tensor_index
        else:
            # Just an integrity check. This should never happen.
            _value_error(state, "available_space cannot be a negative number")

    if total_size_of_tensors != total_size_of_files:
        # Just an integrity check. This should never happen.
        _value_error(state, "Tensor and file sizes are not the same")

    metadata_file_path: str
    if minio_folder_path is None:
        metadata_file_path = _get_full_file_path(
            state, server_round, METADATA_FILE_NAME
        )
    else:
        metadata_file_path = f"{minio_folder_path}/{METADATA_FILE_NAME}"

    metadata_bytes = json.dumps(
        {
            "tensorType": parameters.tensor_type,
            "tensorSizes": tensor_sizes,
            "files": file_list,
        }
    ).encode("utf-8")

    return _push_single_file(state, metadata_file_path, metadata_bytes)


def pull_server_state(
    minio_state: MinioState, server_round: int
) -> ServerStateWithGlobalModel:
    """Pull the server state from MinIO."""
    params_folder_path = _get_params_folder_path(minio_state, server_round)
    result = _pull_single_file(
        state=minio_state,
        full_file_path=f"{params_folder_path}/{SERVER_STATE_FILE}",
    )
    if not isinstance(result, bytes):
        message = f"🚨 Failed to pull state from MinIO (server_round: {server_round})"
        _log_error(minio_state, message)
        raise ConnectionError(message)

    server_id: str
    json_round: int
    contains_momentum: bool
    elapsed_time_in_seconds: float

    try:
        state_json_string = result.decode("utf-8")
        json_root = json.loads(state_json_string)
        server_id = json_root["id"]
        json_round = int(json_root["round"])
        contains_momentum = bool(json_root["contains_momentum"])
        elapsed_time_in_seconds = float(json_root["elapsed_time_in_seconds"])
    except Exception as ex:
        message = f"🚨 Failed to deserialize {SERVER_STATE_FILE}"
        f" (round: {round})"
        _log_error(minio_state, message)
        raise TypeError(message) from ex

    if round != json_round:
        message = f"🚨 Unexpected round number in {SERVER_STATE_FILE}"
        f" (expected: {round}; actual: {json_round})"
        _log_error(minio_state, message)
        raise ValueError(message)

    params_folder_path = _get_params_folder_path(minio_state, json_round)
    global_model = pull_parameters(
        state=minio_state,
        server_round=json_round,
        minio_folder_path=f"{params_folder_path}/{SERVER_GLOBAL_MODEL_FOLDER}",
    )
    if not isinstance(global_model, list):
        message = "🚨 Failed to pull global model from MinIO"
        f" (round: {minio_state.server_round})"
        _log_error(minio_state, message)
        raise ConnectionError(message)

    momentum: NDArrays | None = None
    if contains_momentum:
        params_folder_path = _get_params_folder_path(minio_state, json_round)
        momentum = pull_parameters(
            state=minio_state,
            server_round=json_round,
            minio_folder_path=f"{params_folder_path}/{SERVER_MOMENTUM_FOLDER}",
        )
        if not isinstance(momentum, list):
            message = "🚨 Failed to pull momentum from MinIO"
            f" (round: {minio_state.server_round})"
            _log_error(minio_state, message)
            raise ConnectionError(message)

    history: History
    try:
        params_folder_path = _get_params_folder_path(minio_state, json_round)
        history_bytes = _pull_single_file(
            state=minio_state,
            full_file_path=f"{params_folder_path}/{SERVER_HISTORY_FILE}",
        )
        if isinstance(history_bytes, bytes):
            history = pickle.loads(history_bytes)
        else:
            raise TypeError("Failed to deserialize history")  # noqa: TRY301
    except Exception as ex:
        message = f"🚨 Failed to pull history from MinIO (round: {json_round})"
        _log_error(minio_state, message)
        raise ConnectionError(message) from ex

    return ServerStateWithGlobalModel(
        server_id, json_round, global_model, momentum, elapsed_time_in_seconds, history
    )


def push_server_state(minio_state: MinioState, server_state: ServerState) -> bool:
    """Push the server state to MinIO."""
    # Use the server state as ground truth
    minio_state.endpoint_id = server_state.id

    if isinstance(server_state.momentum, list):
        params_folder_path = _get_params_folder_path(minio_state, server_state.round)
        result = push_parameters(
            state=minio_state,
            server_round=server_state.round,
            parameters=server_state.momentum,
            minio_folder_path=f"{params_folder_path}/{SERVER_MOMENTUM_FOLDER}",
        )
        if not result:
            message = (
                f"🚨 Failed to push momentum to MinIO (round: {server_state.round})"
            )
            _log_error(minio_state, message)
            raise ConnectionError(message)

    params_folder_path = _get_params_folder_path(minio_state, server_state.round)
    result = _push_single_file(
        state=minio_state,
        full_file_path=f"{params_folder_path}/{SERVER_HISTORY_FILE}",
        file_content=pickle.dumps(server_state.history),
    )
    if not result:
        message = "🚨 Failed to push history to MinIO"
        f" (round: {server_state.round})"
        _log_error(minio_state, message)
        raise ConnectionError(message)

    result = _push_single_file(
        state=minio_state,
        full_file_path=f"{params_folder_path}/{SERVER_STATE_FILE}",
        file_content=server_state.to_json().encode("utf-8"),
    )
    if not result:
        message = f"🚨 Failed to push state to MinIO (round: {server_state.round})"
        _log_error(minio_state, message)
        raise ConnectionError(message)

    return True


def _log_info(state: MinioState, message: str) -> None:
    if callable(state.log):
        state.log(INFO, message)


def _log_error(
    state: MinioState, message: str, exception: Exception | None = None
) -> None:
    if callable(state.log):
        state.log(ERROR, message, exc_info=exception, stack_info=True)


def _type_error(
    state: MinioState, message: str, exception: Exception | None = None
) -> None:
    if state.throw_on_error:
        if callable(state.log):
            state.log(ERROR, message)
        raise TypeError(message) from exception if exception else TypeError(message)


def _value_error(
    state: MinioState, message: str, parent_exception: Exception | None = None
) -> None:
    if state.throw_on_error:
        if callable(state.log):
            state.log(ERROR, message)
        raise ValueError(message) from (
            parent_exception if parent_exception else ValueError(message)
        )


def _connection_error(state: MinioState, message: str) -> None:
    if state.throw_on_error:
        if callable(state.log):
            state.log(ERROR, message)
        raise ConnectionError(message)
