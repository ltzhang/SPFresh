# ANN Data Converter for SPFresh

This Python program converts various ANN (Approximate Nearest Neighbor) data formats to binary format compatible with SPFresh.

## Supported Input Formats

### TEXMEX Formats
- **fvecs**: Float vectors (4 + d×4 bytes per vector)
  - Format: `[dimension (int), components (float)×d]`
  - Used in: SIFT datasets, GIST datasets
- **ivecs**: Integer vectors (4 + d×4 bytes per vector)
  - Format: `[dimension (int), components (int)×d]`
  - Used in: Groundtruth files
- **bvecs**: Byte vectors (4 + d bytes per vector)
  - Format: `[dimension (int), components (unsigned char)×d]`
  - Used in: Large SIFT datasets (1B vectors)

### HDF5 Format
- Generic HDF5 files with vector datasets
- Automatically detects suitable datasets

## Output Format

The binary output format includes:
- **Header** (optional): `[num_vectors (int), dimension (int), data_type (int)]`
  - data_type: 0=float32, 1=int32, 2=uint8
- **Data**: Vectors stored row-wise as raw bytes

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Convert Single File

```bash
# Convert fvecs file (auto-detects format, saves to current directory)
python ann_data_converter.py sift/sift_base.fvecs

# Convert with custom output directory
python ann_data_converter.py sift/sift_base.fvecs --output-dir output/

# Convert HDF5 with specific dataset
python ann_data_converter.py mnist-784-euclidean.hdf5 --dataset train --output-dir output/

# Skip confirmation prompt
python ann_data_converter.py sift/sift_base.fvecs --yes
```

### Batch Conversion

```bash
# Convert all fvecs files in a directory
python ann_data_converter.py sift/ --batch --output-dir output/

# Convert all HDF5 files in a directory
python ann_data_converter.py . --batch --format hdf5 --output-dir output/
```

## Examples

### Converting SIFT Dataset

```bash
# Convert base vectors (auto-detects format)
python ann_data_converter.py sift/sift_base.fvecs

# Convert query vectors
python ann_data_converter.py sift/sift_query.fvecs

# Convert learning vectors
python ann_data_converter.py sift/sift_learn.fvecs

# Convert groundtruth
python ann_data_converter.py sift/sift_groundtruth.ivecs

# Convert all to specific output directory
python ann_data_converter.py sift/ --batch --output-dir output/
```

### Converting MNIST HDF5

```bash
# Convert all datasets
python ann_data_converter.py mnist-784-euclidean.hdf5

# Convert specific dataset
python ann_data_converter.py mnist-784-euclidean.hdf5 --dataset train
```

## File Format Details

### TEXMEX Formats (from http://corpus-texmex.irisa.fr/)

- **fvecs**: Little-endian format with 4-byte dimension header followed by float components
- **ivecs**: Little-endian format with 4-byte dimension header followed by int components  
- **bvecs**: Little-endian format with 4-byte dimension header followed by unsigned char components

### Binary Output Format

```
Header (8 bytes):
- num_vectors: int32 (4 bytes)
- dimension: int32 (4 bytes)

Data:
- vectors stored row-wise as raw bytes
- data type is determined by the input format:
  - fvecs: float32 (4 bytes per component)
  - ivecs: int32 (4 bytes per component)
  - bvecs: uint8 (1 byte per component)
  - HDF5: depends on dataset type
```

## Error Handling

The program includes error handling for:
- Invalid file formats
- Corrupted data
- Missing datasets in HDF5 files
- File I/O errors

## Performance Notes

- For large files (>1GB), the program reads data in chunks to manage memory
- HDF5 files are read efficiently using the h5py library
- Binary output is optimized for fast reading in SPFresh

