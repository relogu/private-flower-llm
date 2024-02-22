import io
from logging import DEBUG
import hashlib
import binascii
from minio.helpers import ObjectWriteResult
from flwr.common import log, NDArrays, ndarrays_to_parameters, parameters_to_ndarrays, Parameters
import numpy as np
from pollen_worker.node_manager.minio_state import MinioState
import json

#TODO: Add logging

class MinioTools(object):

    METADATA_FILE_NAME = "metadata.json"

    @staticmethod
    def get_full_file_path(state: MinioState, file_name: str) -> str:
        return f"{MinioTools.get_params_folder_path(state)}/{file_name}"

    @staticmethod
    def get_params_folder_path(state: MinioState) -> str:
        return f"{state.run_uuid}/{MinioTools.get_justified_number(state.server_round)}/{state.node_manager_uuid}"

    @staticmethod
    def get_justified_number(number: int, digits = 8) -> str:
        return str(number).rjust(digits, "0")

    @staticmethod
    def _pull_single_file(state: MinioState, file_list: list, file_index: int) -> bytes:
        file_name = file_list[file_index]["name"]
        file_hash = file_list[file_index]["sha3_256"]
        if len(file_name) < 15:
            raise TypeError("The metadata JSON contains an invalid file name")
        if len(file_hash) != 64:
            raise TypeError("The metadata JSON contains an invalid hash code")
        full_file_path = MinioTools.get_full_file_path(state, file_name)
        file_bytes: bytes = b""
        try:
            response = state.client.get_object(state.bucket_name, full_file_path)
            file_bytes = response.read()
        finally:
            response.close()
            response.release_conn()
        hash = hashlib.sha3_256()
        hash.update(file_bytes)
        actual_file_hash = binascii.hexlify(hash.digest()).decode("utf-8")
        if file_hash != actual_file_hash:
            raise TypeError(f"The computed hash code of file '{full_file_path}' differs from the one stored in metadata.json")
        return file_bytes

    @staticmethod
    def pull_parameters(state: MinioState) -> NDArrays:
        metadata_file_path = MinioTools.get_full_file_path(state, MinioTools.METADATA_FILE_NAME)
        tensor_type = ""
        tensor_sizes = []
        file_list = []
        try:
            response = state.client.get_object(state.bucket_name, metadata_file_path)
            metadata_json_string = response.read().decode("utf-8")
            json_root = json.loads(metadata_json_string)
            tensor_type = json_root["tensorType"]
            tensor_sizes = json_root["tensorSizes"]
            file_list = json_root["files"]
        finally:
            response.close()
            response.release_conn()
        if len(tensor_sizes) == 0 or len(file_list) == 0:
            raise ConnectionError("Failed to pull the metadata file from MinIO")

        tensors: list[bytes] = []
        current_file_index = 0
        current_file_content = b""
        current_file_pointer = 0
        current_tensor_index = 0
        current_tensor_content = bytearray()
        all_files_processed = False
        all_done = False

        current_file_content = MinioTools._pull_single_file(state, file_list, current_file_index)
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
                        current_file_content = MinioTools._pull_single_file(state, file_list, current_file_index)
                        total_size_of_files += len(current_file_content)
                else:
                    current_tensor_content.extend(current_file_content[current_file_pointer: current_file_pointer + available_space])
                    current_file_pointer += available_space
                all_files_processed = len(file_list) <= current_file_index
            else:
                raise ValueError("available_space cannot be a negative number")

        if total_size_of_tensors != total_size_of_files:
            raise ValueError("Tensor and file sizes are not the same")

        return parameters_to_ndarrays(Parameters(tensors, tensor_type))

    @staticmethod
    def push_parameters(state: MinioState, parameters: NDArrays | Parameters) -> bool:
        if isinstance(parameters, list):
            parameters = ndarrays_to_parameters(parameters)
        else: 
            if not isinstance(parameters, Parameters):
                raise TypeError("parameters are not an instance of List (i.e., NDArrays) or Parameters")

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
                current_file_name = f"{MinioTools.get_justified_number(current_file_id)}.params"
                hash = hashlib.sha3_256()
                hash.update(current_file_content)
                current_file_hash = binascii.hexlify(hash.digest()).decode("utf-8")
                file_list.append({
                    "name": current_file_name,
                    "sha3_256": current_file_hash
                })
                full_file_path = MinioTools.get_full_file_path(state, current_file_name)
                write_result = state.client.put_object(
                    state.bucket_name, full_file_path, io.BytesIO(current_file_content), length=len(current_file_content)
                )
                if not (isinstance(write_result, ObjectWriteResult) and len(write_result.object_name) > 0):
                    raise ConnectionError("Failed to push the file to MinIO")
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
                raise ValueError("available_space cannot be a negative number")
        
        if total_size_of_tensors != total_size_of_files:
            raise ValueError("Tensor and file sizes are not the same")

        metadata_file_path = MinioTools.get_full_file_path(state, MinioTools.METADATA_FILE_NAME)
        json_root = {"tensorType": parameters.tensor_type, "tensorSizes": tensor_sizes, "files": file_list}
        metadata_json_string = json.dumps(json_root)
        metadata_bytes = metadata_json_string.encode("utf-8")

        write_result = state.client.put_object(
            state.bucket_name, metadata_file_path, io.BytesIO(metadata_bytes), length=len(metadata_bytes)
        )
        if not (isinstance(write_result, ObjectWriteResult) and len(write_result.object_name) > 0):
            raise ConnectionError("Failed to push the file to MinIO")

        return True
