#!/usr/bin/env python3
"""
ANN Data Converter for SPFresh

This program converts various ANN data formats to binary format:
- fvecs: float vectors (4 + d*4 bytes per vector)
- ivecs: integer vectors (4 + d*4 bytes per vector) 
- bvecs: byte vectors (4 + d bytes per vector)
- HDF5: various formats

Output: Binary format compatible with SPFresh
"""

import struct
import numpy as np
import h5py
import argparse
import os
from pathlib import Path
from typing import Union, Tuple, Optional


def read_fvecs(filename: str) -> np.ndarray:
    """
    Read fvecs file format.
    Each vector takes 4 + d*4 bytes where d is the dimension.
    Format: [dimension (int), components (float)*d]
    """
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            # Read dimension (4 bytes, little endian int)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            
            dim = struct.unpack('<i', dim_bytes)[0]
            
            # Read vector components (dim * 4 bytes, little endian float)
            vec_bytes = f.read(dim * 4)
            if len(vec_bytes) != dim * 4:
                break
                
            vector = struct.unpack(f'<{dim}f', vec_bytes)
            vectors.append(vector)
    
    return np.array(vectors, dtype=np.float32)


def read_ivecs(filename: str) -> np.ndarray:
    """
    Read ivecs file format.
    Each vector takes 4 + d*4 bytes where d is the dimension.
    Format: [dimension (int), components (int)*d]
    """
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            # Read dimension (4 bytes, little endian int)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            
            dim = struct.unpack('<i', dim_bytes)[0]
            
            # Read vector components (dim * 4 bytes, little endian int)
            vec_bytes = f.read(dim * 4)
            if len(vec_bytes) != dim * 4:
                break
                
            vector = struct.unpack(f'<{dim}i', vec_bytes)
            vectors.append(vector)
    
    return np.array(vectors, dtype=np.int32)


def read_bvecs(filename: str) -> np.ndarray:
    """
    Read bvecs file format.
    Each vector takes 4 + d bytes where d is the dimension.
    Format: [dimension (int), components (unsigned char)*d]
    """
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            # Read dimension (4 bytes, little endian int)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            
            dim = struct.unpack('<i', dim_bytes)[0]
            
            # Read vector components (dim bytes, unsigned char)
            vec_bytes = f.read(dim)
            if len(vec_bytes) != dim:
                break
                
            vector = struct.unpack(f'<{dim}B', vec_bytes)
            vectors.append(vector)
    
    return np.array(vectors, dtype=np.uint8)


def read_hdf5(filename: str, dataset_name: Optional[str] = None) -> np.ndarray:
    """
    Read HDF5 file format.
    If dataset_name is not specified, tries to find the first suitable dataset.
    """
    with h5py.File(filename, 'r') as f:
        if dataset_name is None:
            # Find the first dataset that looks like vectors
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    dataset_name = key
                    break
        
        if dataset_name is None or dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file")
        
        dataset = f[dataset_name]
        return dataset[:]


def write_binary_format(data: np.ndarray, output_file: str, 
                       data_type: str = 'float32', 
                       include_metadata: bool = True) -> None:
    """
    Write data to binary format compatible with SPFresh.
    
    Format:
    - Header (optional): [num_vectors (int), dimension (int), data_type (int)]
    - Data: vectors stored row-wise as raw bytes
    """
    num_vectors, dimension = data.shape
    
    with open(output_file, 'wb') as f:
        if include_metadata:
            # Write header: num_vectors, dimension, data_type
            # data_type: 0=float32, 1=int32, 2=uint8
            type_map = {'float32': 0, 'int32': 1, 'uint8': 2}
            data_type_code = type_map.get(data_type, 0)
            
            header = struct.pack('<iii', num_vectors, dimension, data_type_code)
            f.write(header)
        
        # Write data as raw bytes
        f.write(data.tobytes())
    
    print(f"Written {num_vectors} vectors of dimension {dimension} to {output_file}")
    print(f"Data type: {data.dtype}, File size: {os.path.getsize(output_file)} bytes")


def convert_file(input_file: str, output_file: str, 
                input_format: str, dataset_name: Optional[str] = None) -> None:
    """
    Convert a single file from input format to binary format.
    """
    print(f"Converting {input_file} to {output_file}")
    
    # Read input file
    if input_format.lower() == 'fvecs':
        data = read_fvecs(input_file)
        data_type = 'float32'
    elif input_format.lower() == 'ivecs':
        data = read_ivecs(input_file)
        data_type = 'int32'
    elif input_format.lower() == 'bvecs':
        data = read_bvecs(input_file)
        data_type = 'uint8'
    elif input_format.lower() == 'hdf5':
        data = read_hdf5(input_file, dataset_name)
        data_type = str(data.dtype)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")
    
    print(f"Read {data.shape[0]} vectors of dimension {data.shape[1]}")
    
    # Write output file
    write_binary_format(data, output_file, data_type)


def main():
    parser = argparse.ArgumentParser(description='Convert ANN data files to binary format')
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('output_file', help='Output binary file path')
    parser.add_argument('--format', '-f', required=True, 
                       choices=['fvecs', 'ivecs', 'bvecs', 'hdf5'],
                       help='Input file format')
    parser.add_argument('--dataset', '-d', 
                       help='Dataset name for HDF5 files (optional)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process multiple files in a directory')
    
    args = parser.parse_args()
    
    if args.batch and os.path.isdir(args.input_file):
        # Process all files in directory
        input_dir = Path(args.input_file)
        output_dir = Path(args.output_file)
        output_dir.mkdir(exist_ok=True)
        
        # Find files with the specified format
        if args.format == 'hdf5':
            pattern = '*.hdf5'
        else:
            pattern = f'*.{args.format}'
        
        for input_file in input_dir.glob(pattern):
            output_file = output_dir / f"{input_file.stem}.bin"
            try:
                convert_file(str(input_file), str(output_file), args.format, args.dataset)
            except Exception as e:
                print(f"Error converting {input_file}: {e}")
    else:
        # Process single file
        convert_file(args.input_file, args.output_file, args.format, args.dataset)


if __name__ == "__main__":
    main()
