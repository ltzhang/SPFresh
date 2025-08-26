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


def read_hdf5(filename: str, dataset_name: Optional[str] = None) -> Union[np.ndarray, dict]:
    """
    Read HDF5 file format.
    If dataset_name is not specified, returns all datasets as a dictionary.
    If dataset_name is specified, returns only that dataset as numpy array.
    """
    with h5py.File(filename, 'r') as f:
        if dataset_name is None:
            # Return all datasets
            datasets = {}
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    datasets[key] = f[key][:]
            return datasets
        else:
            # Return specific dataset
            if dataset_name not in f:
                raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file")
            dataset = f[dataset_name]
            return dataset[:]


def get_hdf5_datasets(filename: str) -> list:
    """
    Get list of available dataset names in HDF5 file.
    """
    with h5py.File(filename, 'r') as f:
        return [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]


def write_binary_format(data: np.ndarray, output_file: str, 
                       data_type: str = 'float32', 
                       include_metadata: bool = True) -> None:
    """
    Write data to binary format compatible with SPFresh.
    
    Format:
    - Header (optional): [num_vectors (int), dimension (int)]
    - Data: vectors stored row-wise as raw bytes
    """
    num_vectors, dimension = data.shape
    
    with open(output_file, 'wb') as f:
        if include_metadata:
            # Write header: num_vectors, dimension (8 bytes total)
            header = struct.pack('<ii', num_vectors, dimension)
            f.write(header)
        
        # Write data as raw bytes
        f.write(data.tobytes())
    
    print(f"Written {num_vectors} vectors of dimension {dimension} to {output_file}")
    print(f"Data type: {data.dtype}, File size: {os.path.getsize(output_file)} bytes")


def generate_output_filename(input_file: str, output_dir: str, 
                           input_format: str, dataset_name: str = None,
                           suffix: str = None, data_type: str = None) -> str:
    """
    Generate appropriate output filename based on input file and format.
    
    For TEXMEX formats: adds data type suffix + .ubin
    For HDF5: uses dataset name + data type suffix + .ubin
    """
    input_path = Path(input_file)
    input_stem = input_path.stem
    
    # Determine data type suffix
    if data_type:
        if data_type == 'float32':
            type_suffix = 'f32'
        elif data_type == 'int32':
            type_suffix = 'i32'
        elif data_type == 'uint8':
            type_suffix = 'u8'
        else:
            # For other types, use a generic suffix
            type_suffix = data_type.replace('float', 'f').replace('int', 'i').replace('uint', 'u')
    else:
        # Default type suffix based on input format
        if input_format.lower() == 'fvecs':
            type_suffix = 'f32'
        elif input_format.lower() == 'ivecs':
            type_suffix = 'i32'
        elif input_format.lower() == 'bvecs':
            type_suffix = 'u8'
        else:
            type_suffix = 'bin'
    
    if input_format.lower() == 'hdf5' and dataset_name:
        # For HDF5, use dataset name + type suffix
        output_name = f"{input_stem}_{dataset_name}_{type_suffix}.ubin"
    elif suffix:
        # Use custom suffix + type suffix
        output_name = f"{input_stem}_{suffix}_{type_suffix}.ubin"
    else:
        # Default: add type suffix
        output_name = f"{input_stem}_{type_suffix}.ubin"
    
    return os.path.join(output_dir, output_name)


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
        print(f"Read {data.shape[0]} vectors of dimension {data.shape[1]}")
        # Write output file
        write_binary_format(data, output_file, data_type)
        
    elif input_format.lower() == 'ivecs':
        data = read_ivecs(input_file)
        data_type = 'int32'
        print(f"Read {data.shape[0]} vectors of dimension {data.shape[1]}")
        # Write output file
        write_binary_format(data, output_file, data_type)
        
    elif input_format.lower() == 'bvecs':
        data = read_bvecs(input_file)
        data_type = 'uint8'
        print(f"Read {data.shape[0]} vectors of dimension {data.shape[1]}")
        # Write output file
        write_binary_format(data, output_file, data_type)
        
    elif input_format.lower() == 'hdf5':
        if dataset_name is None:
            # Convert all datasets
            datasets = read_hdf5(input_file)
            print(f"Found {len(datasets)} datasets in HDF5 file")
            
            for name, data in datasets.items():
                if len(data.shape) == 1:
                    # 1D data, skip
                    print(f"Skipping 1D dataset '{name}' with shape {data.shape}")
                    continue
                    
                if len(data.shape) > 2:
                    # Reshape higher dimensional data to 2D
                    data = data.reshape(data.shape[0], -1)
                
                # Generate output filename for this dataset
                data_type = str(data.dtype)
                dataset_output_file = generate_output_filename(input_file, os.path.dirname(output_file), 
                                                           'hdf5', name, data_type=data_type)
                
                print(f"Converting dataset '{name}': {data.shape[0]} vectors of dimension {data.shape[1]}")
                write_binary_format(data, dataset_output_file, data_type)
        else:
            # Convert specific dataset
            data = read_hdf5(input_file, dataset_name)
            if len(data.shape) == 1:
                raise ValueError(f"Dataset '{dataset_name}' is 1D, expected 2D data")
            if len(data.shape) > 2:
                data = data.reshape(data.shape[0], -1)
                
            data_type = str(data.dtype)
            print(f"Read {data.shape[0]} vectors of dimension {data.shape[1]}")
            write_binary_format(data, output_file, data_type)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")


def detect_format_from_filename(filename: str) -> str:
    """
    Detect input format from file extension.
    Returns the detected format or None if unknown.
    """
    ext = Path(filename).suffix.lower()
    
    if ext == '.fvecs':
        return 'fvecs'
    elif ext == '.ivecs':
        return 'ivecs'
    elif ext == '.bvecs':
        return 'bvecs'
    elif ext in ['.hdf5', '.h5']:
        return 'hdf5'
    else:
        return None


def print_file_info(input_file: str, input_format: str, dataset_name: Optional[str] = None) -> None:
    """
    Print detailed information about the input file.
    """
    print(f"\n📁 Input File Information:")
    print(f"  File: {input_file}")
    print(f"  Format: {input_format}")
    
    # Get file size
    try:
        file_size = os.path.getsize(input_file)
        if file_size > 1024**3:
            size_str = f"{file_size / (1024**3):.2f} GB"
        elif file_size > 1024**2:
            size_str = f"{file_size / (1024**2):.2f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.2f} KB"
        else:
            size_str = f"{file_size} bytes"
        print(f"  Size: {size_str} ({file_size:,} bytes)")
    except Exception as e:
        print(f"  Size: Error reading file size: {e}")
    
    # Get data information
    try:
        if input_format in ['fvecs', 'ivecs', 'bvecs']:
            # For TEXMEX formats, read header info
            if input_format == 'fvecs':
                data = read_fvecs(input_file)
            elif input_format == 'ivecs':
                data = read_ivecs(input_file)
            elif input_format == 'bvecs':
                data = read_bvecs(input_file)
            
            print(f"  📊 Data Statistics:")
            print(f"    Vectors: {data.shape[0]:,}")
            print(f"    Dimensions: {data.shape[1]}")
            print(f"    Data type: {data.dtype}")
            
            # Calculate memory usage
            if data.dtype == np.float32:
                bytes_per_vector = data.shape[1] * 4
            elif data.dtype == np.int32:
                bytes_per_vector = data.shape[1] * 4
            elif data.dtype == np.uint8:
                bytes_per_vector = data.shape[1]
            else:
                bytes_per_vector = data.shape[1] * 4  # default assumption
            
            total_data_bytes = data.shape[0] * bytes_per_vector
            print(f"    Memory per vector: {bytes_per_vector} bytes")
            print(f"    Total data size: {total_data_bytes:,} bytes")
            print(f"    Estimated output size: {total_data_bytes + 8:,} bytes (with 8-byte header)")
            
        elif input_format == 'hdf5':
            datasets = get_hdf5_datasets(input_file)
            print(f"  📊 HDF5 Information:")
            print(f"    Available datasets: {len(datasets)}")
            for i, name in enumerate(datasets):
                try:
                    dataset_data = read_hdf5(input_file, name)
                    if len(dataset_data.shape) == 1:
                        print(f"    Dataset '{name}': 1D array, {dataset_data.shape[0]:,} elements")
                    else:
                        print(f"    Dataset '{name}': {dataset_data.shape[0]:,} vectors × {dataset_data.shape[1]} dimensions")
                        print(f"      Data type: {dataset_data.dtype}")
                        if len(dataset_data.shape) == 2:
                            bytes_per_vector = dataset_data.shape[1] * 4  # assuming 4 bytes per element
                            total_bytes = dataset_data.shape[0] * bytes_per_vector
                            print(f"      Estimated output size: {total_bytes + 8:,} bytes")
                except Exception as e:
                    print(f"    Dataset '{name}': Error reading: {e}")
            
            if dataset_name:
                print(f"\n  🎯 Selected Dataset: {dataset_name}")
                data = read_hdf5(input_file, dataset_name)
                if len(data.shape) == 1:
                    print(f"    Type: 1D array")
                    print(f"    Elements: {data.shape[0]:,}")
                else:
                    print(f"    Vectors: {data.shape[0]:,}")
                    print(f"    Dimensions: {data.shape[1]}")
                    print(f"    Data type: {data.dtype}")
                    if len(data.shape) == 2:
                        bytes_per_vector = data.shape[1] * 4
                        total_bytes = data.shape[0] * bytes_per_vector
                        print(f"    Estimated output size: {total_bytes + 8:,} bytes")
                
    except Exception as e:
        print(f"  ❌ Error reading file data: {e}")
    
    print()


def confirm_conversion(input_file: str, output_file: str, input_format: str, 
                      dataset_name: Optional[str] = None) -> bool:
    """
    Ask user to confirm the conversion details.
    """
    print(f"\n🔄 Conversion Details:")
    print(f"  Input file: {input_file}")
    print(f"  Input format: {input_format}")
    print(f"  Output file: {output_file}")
    if dataset_name:
        print(f"  Dataset: {dataset_name}")
    
    print()
    response = input("Proceed with conversion? (y/N): ").strip().lower()
    return response in ['y', 'yes']


def main():
    parser = argparse.ArgumentParser(description='Convert ANN data files to binary format')
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('--format', '-f', 
                       choices=['fvecs', 'ivecs', 'bvecs', 'hdf5'],
                       help='Input file format (auto-detected if not specified)')
    parser.add_argument('--dataset', '-d', 
                       help='Dataset name for HDF5 files (optional)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process multiple files in a directory')
    parser.add_argument('--list-datasets', '-l', action='store_true',
                       help='List available datasets in HDF5 file (for HDF5 format only)')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')
    parser.add_argument('--output-dir', '-o', default='.',
                       help='Output directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Auto-detect format if not specified
    if args.format is None:
        if args.batch and os.path.isdir(args.input_file):
            # For batch processing, try to detect format from files in directory
            input_dir = Path(args.input_file)
            for ext in ['.fvecs', '.ivecs', '.bvecs', '.hdf5', '.h5']:
                if list(input_dir.glob(f'*{ext}')):
                    args.format = detect_format_from_filename(f'dummy{ext}')
                    break
        else:
            # For single file, detect from filename
            args.format = detect_format_from_filename(args.input_file)
        
        if args.format is None:
            print(f"Error: Could not auto-detect format from files in: {args.input_file}")
            print("Please specify format using --format option")
            return
        print(f"Auto-detected format: {args.format}")
    
    # Handle HDF5 dataset listing
    if args.list_datasets and args.format == 'hdf5':
        if not os.path.isfile(args.input_file):
            print(f"Error: {args.input_file} is not a file")
            return
        
        try:
            datasets = get_hdf5_datasets(args.input_file)
            print(f"Available datasets in {args.input_file}:")
            for i, name in enumerate(datasets):
                print(f"  {i+1}. {name}")
            return
        except Exception as e:
            print(f"Error reading HDF5 file: {e}")
            return
    
    if args.batch and os.path.isdir(args.input_file):
        # Process all files in directory
        input_dir = Path(args.input_file)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Find files with the specified format
        if args.format == 'hdf5':
            pattern = '*.hdf5'
        else:
            pattern = f'*.{args.format}'
        
        files_to_convert = list(input_dir.glob(pattern))
        print(f"Found {len(files_to_convert)} files to convert:")
        for f in files_to_convert:
            print(f"  {f}")
        
        if not args.yes:
            response = input(f"\nProceed with batch conversion of {len(files_to_convert)} files? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Batch conversion cancelled.")
                return
        
        for input_file in files_to_convert:
            try:
                if args.format == 'hdf5':
                    # For HDF5, output_dir is used directly in convert_file
                    convert_file(str(input_file), str(output_dir), args.format, args.dataset)
                else:
                    # For other formats, generate output filename
                    output_file = generate_output_filename(str(input_file), str(output_dir), args.format)
                    convert_file(str(input_file), output_file, args.format, args.dataset)
            except Exception as e:
                print(f"Error converting {input_file}: {e}")
    else:
        # Process single file
        if args.format == 'hdf5' and args.dataset is None:
            # For HDF5 without specific dataset, output_dir is used
            output_dir = args.output_dir
            # Print file information
            print_file_info(args.input_file, args.format, args.dataset)
            if not args.yes:
                if not confirm_conversion(args.input_file, output_dir, args.format):
                    print("Conversion cancelled.")
                    return
            convert_file(args.input_file, output_dir, args.format, args.dataset)
        else:
            # For other formats or HDF5 with specific dataset, generate output filename
            if args.format == 'hdf5':
                # For HDF5 with specific dataset, we need to read the data first to get the type
                # Then generate filename and convert
                data = read_hdf5(args.input_file, args.dataset)
                if len(data.shape) == 1:
                    raise ValueError(f"Dataset '{args.dataset}' is 1D, expected 2D data")
                if len(data.shape) > 2:
                    data = data.reshape(data.shape[0], -1)
                data_type = str(data.dtype)
                output_file = generate_output_filename(args.input_file, args.output_dir, 
                                                    args.format, args.dataset, data_type=data_type)
                # Print file information
                print_file_info(args.input_file, args.format, args.dataset)
                if not args.yes:
                    if not confirm_conversion(args.input_file, output_file, args.format, args.dataset):
                        print("Conversion cancelled.")
                        return
                convert_file(args.input_file, output_file, args.format, args.dataset)
            else:
                # For other formats, generate output filename
                output_file = generate_output_filename(args.input_file, args.output_dir, 
                                                    args.format, args.dataset)
                # Print file information
                print_file_info(args.input_file, args.format, args.dataset)
                if not args.yes:
                    if not confirm_conversion(args.input_file, output_file, args.format, args.dataset):
                        print("Conversion cancelled.")
                        return
                convert_file(args.input_file, output_file, args.format, args.dataset)


if __name__ == "__main__":
    main()

