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
# Convert fvecs file
python ann_data_converter.py sift/sift_base.fvecs sift_base.bin --format fvecs

# Convert ivecs file
python ann_data_converter.py sift/sift_groundtruth.ivecs groundtruth.bin --format ivecs

# Convert HDF5 file
python ann_data_converter.py mnist-784-euclidean.hdf5 mnist.bin --format hdf5

# Convert HDF5 with specific dataset
python ann_data_converter.py mnist-784-euclidean.hdf5 mnist.bin --format hdf5 --dataset train
```

### Batch Conversion

```bash
# Convert all fvecs files in a directory
python ann_data_converter.py sift/ output/ --format fvecs --batch

# Convert all HDF5 files in a directory
python ann_data_converter.py . output/ --format hdf5 --batch
```

## Examples

### Converting SIFT Dataset

```bash
# Convert base vectors
python ann_data_converter.py sift/sift_base.fvecs sift_base.bin --format fvecs

# Convert query vectors
python ann_data_converter.py sift/sift_query.fvecs sift_query.bin --format fvecs

# Convert learning vectors
python ann_data_converter.py sift/sift_learn.fvecs sift_learn.bin --format fvecs

# Convert groundtruth
python ann_data_converter.py sift/sift_groundtruth.ivecs sift_groundtruth.bin --format ivecs
```

### Converting MNIST HDF5

```bash
python ann_data_converter.py mnist-784-euclidean.hdf5 mnist.bin --format hdf5
```

## File Format Details

### TEXMEX Formats (from http://corpus-texmex.irisa.fr/)

- **fvecs**: Little-endian format with 4-byte dimension header followed by float components
- **ivecs**: Little-endian format with 4-byte dimension header followed by int components  
- **bvecs**: Little-endian format with 4-byte dimension header followed by unsigned char components

### Binary Output Format

```
Header (12 bytes):
- num_vectors: int32 (4 bytes)
- dimension: int32 (4 bytes) 
- data_type: int32 (4 bytes)

Data:
- vectors stored row-wise as raw bytes
- data_type determines component size:
  - 0: float32 (4 bytes per component)
  - 1: int32 (4 bytes per component)
  - 2: uint8 (1 byte per component)
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
