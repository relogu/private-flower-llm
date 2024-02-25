import gc
import io
from logging import ERROR, INFO
import hashlib
import binascii
from re import template
import time
from minio import S3Error
from minio.helpers import ObjectWriteResult
from flwr.common import NDArrays, ndarrays_to_parameters, parameters_to_ndarrays, Parameters
from pollen_worker.node_manager.minio_state import MinioState
import json

class MinioTools(object):

    METADATA_FILE_NAME = "metadata.json"

    @staticmethod
    def _get_full_file_path(state: MinioState, file_name: str) -> str:
        return f"{MinioTools._get_params_folder_path(state)}/{file_name}"

    @staticmethod
    def _get_params_folder_path(state: MinioState) -> str:
        return f"{state.run_uuid}/{MinioTools._get_justified_number(state.server_round)}/{state.endpoint_id}"

    @staticmethod
    def _get_justified_number(number: int, digits = 8) -> str:
        return str(number).rjust(digits, "0")

    @staticmethod
    def _pull_single_parameters_file(state: MinioState, file_list: list, file_index: int) -> bytes | None:
        file_name: str
        file_hash: str
        try:
            file_name = file_list[file_index]["name"]
            file_hash = file_list[file_index]["sha3_256"]
        except:
            MinioTools._value_error(state, "Failed to retrieve file_name of file_hash from the list")
            return None
        if len(file_name) < 15:
            MinioTools._value_error(state, "The metadata contain an invalid file name")
            return None
        if len(file_hash) != 64:
            MinioTools._value_error(state, "The metadata contain an invalid hash code")
            return None
        full_file_path = MinioTools._get_full_file_path(state, file_name)
        file_bytes = MinioTools._pull_single_file(state, full_file_path)
        if not isinstance(file_bytes, bytes):
            return None
        hash = hashlib.sha3_256()
        hash.update(file_bytes)
        actual_file_hash = binascii.hexlify(hash.digest()).decode("utf-8")
        if file_hash != actual_file_hash:
            MinioTools._value_error(state, f"The computed hash code of file '{full_file_path}' differs from the one stored in metadata.json")
            return None
        return file_bytes

    @staticmethod
    def _pull_single_file(state: MinioState, full_file_path: str) -> bytes | None:
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
                    message = f"🚨 The requested file does not exist on MinIO: {full_file_path}"
                    MinioTools._log_error(state, message)
                    raise ValueError(message)
                else:
                    MinioTools._log_error(state, f"Failed to pull the file: {full_file_path}")
            elapsed_time = time.time() - start_time
            try_again = (not pull_successful) and (elapsed_time < state.timeout_in_seconds)
            if try_again:
                MinioTools._log_info(state, f"Attempt #{number_of_attemts}; trying again to pull the file from MinIO: {full_file_path}")
                gc.collect()
                time.sleep(3)
        if pull_successful and number_of_attemts > 1:
            MinioTools._log_info(state, f"Successfully pulled the file from MinIO after {number_of_attemts} attempts: {full_file_path}")
        if not pull_successful:
            MinioTools._connection_error(state, f"🚨 Timed out after {number_of_attemts} attempt(s) and {int(elapsed_time)} second(s). Completely failed to pull the file from MinIO: {full_file_path}")
            return None
        return file_bytes

    @staticmethod
    def pull_parameters(state: MinioState) -> NDArrays | None:
        tensor_type = ""
        tensor_sizes = []
        file_list = []
        metadata_file_path = MinioTools._get_full_file_path(state, MinioTools.METADATA_FILE_NAME)
        metadata_bytes = MinioTools._pull_single_file(state, metadata_file_path)
        if not isinstance(metadata_bytes, bytes):
            return None
        try:
            metadata_json_string = metadata_bytes.decode("utf-8")
            json_root = json.loads(metadata_json_string)
            tensor_type = json_root["tensorType"]
            tensor_sizes = json_root["tensorSizes"]
            file_list = json_root["files"]
        except:
            MinioTools._type_error(state, f"Failed to deserialize the metadata JSON file: {metadata_file_path}")
            return None
        if len(tensor_sizes) == 0 or len(file_list) == 0:
            MinioTools._value_error(state, "Failed to retrieve values from the metadata JSON file")
            return None

        tensors: list[bytes] = []
        current_file_index = 0
        current_file_content: bytes | None
        current_file_pointer = 0
        current_tensor_index = 0
        current_tensor_content = bytearray()
        all_files_processed = False
        all_done = False

        current_file_content = MinioTools._pull_single_parameters_file(state, file_list, current_file_index)
        if not isinstance(current_file_content, bytes):
            return None
        total_size_of_files = len(current_file_content)
        total_size_of_tensors = 0

        while not all_done:
            available_space = tensor_sizes[current_tensor_index] - len(current_tensor_content)
            if available_space == 0 or all_files_processed:
                tensors.append(bytes(current_tensor_content))
                current_tensor_content.clear()
                total_size_of_tensors += tensor_sizes[current_tensor_index]
                if all_files_processed:
                    all_done = True
                elif current_tensor_index < len(tensor_sizes) -1:
                    current_tensor_index += 1
            elif available_space > 0:
                current_file_size = len(current_file_content) - current_file_pointer
                if available_space >= current_file_size:
                    current_tensor_content.extend(current_file_content[current_file_pointer: len(current_file_content)])
                    current_file_index += 1
                    current_file_pointer = 0
                    if current_file_index < len(file_list):
                        current_file_content = MinioTools._pull_single_parameters_file(state, file_list, current_file_index)
                        if not isinstance(current_file_content, bytes):
                            return None
                        total_size_of_files += len(current_file_content)
                else:
                    current_tensor_content.extend(current_file_content[current_file_pointer: current_file_pointer + available_space])
                    current_file_pointer += available_space
                all_files_processed = len(file_list) <= current_file_index
            else:
                MinioTools._value_error(state, "available_space cannot be a negative number")
                return None

        if total_size_of_tensors != total_size_of_files:
            MinioTools._value_error(state, "Tensor and file sizes are not the same")
            return None

        return parameters_to_ndarrays(Parameters(tensors, tensor_type))

    @staticmethod
    def _push_single_file(state: MinioState, full_file_path: str, file_content: bytes | bytearray) -> bool:
        start_time = time.time()
        elapsed_time = 0.0
        try_again = True
        push_successful = False
        number_of_attemts = 0
        while try_again:
            number_of_attemts += 1
            try:
                write_result = state.client.put_object(
                    state.bucket_name, full_file_path, io.BytesIO(file_content), length = len(file_content)
                )
                if not (isinstance(write_result, ObjectWriteResult) and len(write_result.object_name) > 0):
                    MinioTools._connection_error(state, f"Failed to push the file to MinIO: {full_file_path}")
                push_successful = True
            except:
                MinioTools._log_error(state, f"Failed to push the file to MinIO: {full_file_path}")
            elapsed_time = time.time() - start_time
            try_again = (not push_successful) and (elapsed_time < state.timeout_in_seconds)
            if try_again:
                MinioTools._log_info(state, f"Attempt #{number_of_attemts}; trying again to push the file to MinIO: {full_file_path}")
                gc.collect()
                time.sleep(3)
        if push_successful and number_of_attemts > 1:
            MinioTools._log_info(state, f"Successfully pushed the file to MinIO after {number_of_attemts} attempts: {full_file_path}")
        if not push_successful:
            MinioTools._connection_error(state, f"🚨 Timed out after {number_of_attemts} attempt(s) and {int(elapsed_time)} second(s). Completely failed to push the file to MinIO: {full_file_path}")
        return push_successful

    @staticmethod
    def push_parameters(state: MinioState, parameters: NDArrays | Parameters) -> bool:
        if isinstance(parameters, list):
            parameters = ndarrays_to_parameters(parameters)
        else: 
            if not isinstance(parameters, Parameters):
                MinioTools._type_error(state, "parameters are not an instance of List (i.e., NDArrays) or Parameters")

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
                current_file_name = f"{MinioTools._get_justified_number(current_file_id)}.params"
                hash = hashlib.sha3_256()
                hash.update(current_file_content)
                current_file_hash = binascii.hexlify(hash.digest()).decode("utf-8")
                file_list.append({
                    "name": current_file_name,
                    "sha3_256": current_file_hash
                })
                full_file_path = MinioTools._get_full_file_path(state, current_file_name)

                result = MinioTools._push_single_file(state, full_file_path, current_file_content)
                if not result:
                    return False

                if all_tensors_processed:
                    all_done = True
                else:
                    current_file_id += 1
                total_size_of_files += len(current_file_content)
                current_file_content.clear()
            elif available_space > 0:
                current_tensor_size = len(tensors[current_tensor_index]) - current_tensor_pointer
                if available_space >= current_tensor_size:
                    current_file_content.extend(tensors[current_tensor_index][current_tensor_pointer: len(tensors[current_tensor_index])])
                    tensor_sizes.append(len(tensors[current_tensor_index]))
                    total_size_of_tensors += len(tensors[current_tensor_index])
                    current_tensor_index += 1
                    current_tensor_pointer = 0
                else:
                    current_file_content.extend(tensors[current_tensor_index][current_tensor_pointer: current_tensor_pointer + available_space])
                    current_tensor_pointer += available_space
                all_tensors_processed = len(tensors) == current_tensor_index
            else:
                # Just an integrity check. This should never happen.
                MinioTools._value_error(state, "available_space cannot be a negative number")
        
        if total_size_of_tensors != total_size_of_files:
            # Just a integrity check. This should never happen.
            MinioTools._value_error(state, "Tensor and file sizes are not the same")

        metadata_file_path = MinioTools._get_full_file_path(state, MinioTools.METADATA_FILE_NAME)
        json_root = {"tensorType": parameters.tensor_type, "tensorSizes": tensor_sizes, "files": file_list}
        metadata_json_string = json.dumps(json_root)
        metadata_bytes = metadata_json_string.encode("utf-8")

        return MinioTools._push_single_file(state, metadata_file_path, metadata_bytes)

    @staticmethod
    def _log_info(state: MinioState, message: str) -> None:
        if callable(state.log):
            state.log(INFO, message)

    @staticmethod
    def _log_error(state: MinioState, message: str) -> None:
        if callable(state.log):
            state.log(ERROR, message)

    @staticmethod
    def _type_error(state: MinioState, message: str) -> None:
        if state.throw_on_error:
            if callable(state.log):
                state.log(ERROR, message)
            raise TypeError(message)
        
    @staticmethod
    def _value_error(state: MinioState, message: str) -> None:
        if state.throw_on_error:
            if callable(state.log):
                state.log(ERROR, message)
            raise ValueError(message)

    @staticmethod
    def _connection_error(state: MinioState, message: str) -> None:
        if state.throw_on_error:
            if callable(state.log):
                state.log(ERROR, message)
            raise ConnectionError(message)
