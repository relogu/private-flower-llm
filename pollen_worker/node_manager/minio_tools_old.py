import io
from logging import DEBUG
import hashlib
import binascii
from minio.helpers import ObjectWriteResult
from flwr.common import log, NDArrays, ndarrays_to_parameters, parameters_to_ndarrays, Parameters
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
    def pull_parameters(state: MinioState) -> NDArrays:
        metadata_file_path = MinioTools.get_full_file_path(state, MinioTools.METADATA_FILE_NAME)
        file_list = []
        tensor_type = ""
        try:
            response = state.client.get_object(state.bucket_name, metadata_file_path)
            metadata_json_string = response.read().decode("utf-8")
            json_root = json.loads(metadata_json_string)
            file_list = json_root["files"]
            tensor_type = json_root["tensor_type"]
        finally:
            response.close()
            response.release_conn()
        if len(file_list) == 0:
            raise ConnectionError("Failed to pull the metadata file from MinIO")

        tensors: list[bytes] = []
        number_of_files = len(file_list)
        for index in range(0, number_of_files, 1):
            current_file_name = file_list[index]["name"]
            current_file_hash = file_list[index]["sha3_256"]
            if len(current_file_name) < 15:
                raise TypeError("The metadata JSON contains an invalid file name")
            if len(current_file_hash) != 64:
                raise TypeError("The metadata JSON contains an invalid hash code")
            full_file_path = MinioTools.get_full_file_path(state, current_file_name)
            current_file_bytes: bytes = b""
            try:
                response = state.client.get_object(state.bucket_name, full_file_path)
                current_file_bytes = response.read()
            finally:
                response.close()
                response.release_conn()
            hash = hashlib.sha3_256()
            hash.update(current_file_bytes)
            actual_file_hash = binascii.hexlify(hash.digest()).decode("utf-8")
            if current_file_hash != actual_file_hash:
                raise TypeError(f"The computed hash code of file '{full_file_path}' differs from the one stored in metadata.json")
            tensors_in_file = file_list[index]["tensors"]
            if not (isinstance(tensors_in_file, list) and len(tensors_in_file) > 0):
                raise TypeError("Invalid list of tensors")
            if len(tensors_in_file) == 1:
                tensors.append(current_file_bytes)
            else:
                tensor_start = 0
                for tensor_length in tensors_in_file:
                    chunk = current_file_bytes[tensor_start:tensor_start + tensor_length]
                    tensors.append(chunk)
                    tensor_start += tensor_length

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
        current_file_size = 0
        tensors_in_file = []
        current_file_content = bytearray()
        number_of_tensors = len(tensors)
        for index in range(0, number_of_tensors, 1):
            current_file_size += len(tensors[index])
            tensors_in_file.append(len(tensors[index]))
            current_file_content.extend(tensors[index])
            if (current_file_size >= state.file_size) or (index == number_of_tensors - 1):
                current_file_name = f"{MinioTools.get_justified_number(current_file_id)}.params"
                hash = hashlib.sha3_256()
                hash.update(current_file_content)
                current_file_hash = binascii.hexlify(hash.digest()).decode("utf-8")
                file_list.append({
                    "name": current_file_name,
                    "tensors": tensors_in_file,
                    "sha3_256": current_file_hash
                })
                full_file_path = MinioTools.get_full_file_path(state, current_file_name)
                write_result = state.client.put_object(
                    state.bucket_name, full_file_path, io.BytesIO(current_file_content), length=current_file_size
                )
                if not (isinstance(write_result, ObjectWriteResult) and len(write_result.object_name) > 0):
                    raise ConnectionError("Failed to push the file to MinIO")
                current_file_id += 1
                current_file_size = 0
                tensors_in_file = []
                current_file_content = bytearray()

        metadata_file_path = MinioTools.get_full_file_path(state, MinioTools.METADATA_FILE_NAME)
        json_root = {"files": file_list, "tensor_type": parameters.tensor_type}
        metadata_json_string = json.dumps(json_root)
        metadata_bytes = metadata_json_string.encode("utf-8")

        write_result = state.client.put_object(
            state.bucket_name, metadata_file_path, io.BytesIO(metadata_bytes), length=len(metadata_bytes)
        )
        if not (isinstance(write_result, ObjectWriteResult) and len(write_result.object_name) > 0):
            raise ConnectionError("Failed to push the file to MinIO")

        return True
