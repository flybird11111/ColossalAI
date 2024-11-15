# a python safetensors serializer modified from https://github.com/huggingface/safetensors/blob/41bd1acf38ad28ac559522d40596c6c802f79453/safetensors/src/tensor.rs#L214
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import _TYPES

try:
    from tensornvme.async_file_io import AsyncFileWriter
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install tensornvme to use NVMeOptimizer")
_TYPES_INV = {v: k for k, v in _TYPES.items()}


@dataclass
class TensorInfo:
    dtype: str
    shape: List[int]
    data_offsets: Tuple[int, int]


@dataclass
class PreparedData:
    n: int
    header_bytes: bytes
    offset: int


def flatten_dict(nested_dict, parent_key="", separator="^"):
    """
    Flatten a nested dictionary, generating a flattened dictionary where the keys are joined by the specified separator.

    nested_dict: The input nested dictionary.
    parent_key: The parent key currently being processed.
    separator: The separator used to join keys, default is '_', but can be customized to another symbol. :return: A flattened dictionary."
    """
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, separator).items())
        else:
            items.append((new_key, v))

    return dict(items)


def unflatten_dict(flattened_dict, separator="^"):
    """
    Restore a flattened dictionary back to a multi-level nested dictionary.

    flattened_dict: The flattened dictionary.
    separator: The separator used during flattening, default is '_', but can be customized to another symbol. :return: The restored nested dictionary.
    """
    nested_dict = {}
    for key, value in flattened_dict.items():
        keys = key.split(separator)
        if len(keys) == 1:
            return flattened_dict
        try:
            keys[0] = int(keys[0])
        except ValueError:
            raise (f"{keys[0]} can't convert to integer")
        d = nested_dict
        for part in keys[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        assert isinstance(value, torch.Tensor)
        d[keys[-1]] = value

    return nested_dict


def prepare(data: Dict[str, torch.Tensor]) -> Tuple[PreparedData, List[torch.Tensor], List[str]]:
    sorted_data = sorted(data.items(), key=lambda x: (x[1].dtype, x[0]))

    tensors = []
    tensor_keys = []
    metadata = {}
    offset = 0

    for name, tensor in sorted_data:
        n = tensor.numel() * tensor.element_size()
        tensor_info = TensorInfo(
            dtype=_TYPES_INV[tensor.dtype], shape=list(tensor.shape), data_offsets=(offset, offset + n)
        )
        offset += n
        metadata[name] = asdict(tensor_info)
        tensors.append(tensor)
        tensor_keys.append(name)

    metadata_buf = json.dumps(metadata).encode("utf-8")

    extra = (8 - len(metadata_buf) % 8) % 8
    metadata_buf += b" " * extra

    n = len(metadata_buf)

    return PreparedData(n=n, header_bytes=metadata_buf, offset=offset), tensors, tensor_keys


def save(f_writer: AsyncFileWriter, state_dict: Dict[str, torch.Tensor]) -> None:
    prepared_data, tensors, _ = prepare(state_dict)
    n, header_bytes, _ = prepared_data.n, prepared_data.header_bytes, prepared_data.offset

    f_writer.write(n.to_bytes(8, byteorder="little"))
    f_writer.write(header_bytes)

    for tensor in tensors:
        f_writer.write_raw(tensor, tensor.data_ptr(), tensor.numel() * tensor.element_size(), f_writer.offset)


def move_and_save(
    f_writer: AsyncFileWriter,
    state_dict: Dict[str, torch.Tensor],
    state_dict_pinned: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    prepared_data, _, tensor_keys = prepare(state_dict)
    n, header_bytes, _ = prepared_data.n, prepared_data.header_bytes, prepared_data.offset

    f_writer.write(n.to_bytes(8, byteorder="little"))
    f_writer.write(header_bytes)

    f_writer.register_h2d(len(tensor_keys))
    for name in tensor_keys:
        if state_dict_pinned:
            f_writer.write_tensor(state_dict[name], state_dict_pinned[name])
        else:
            f_writer.write_tensor(state_dict[name])
