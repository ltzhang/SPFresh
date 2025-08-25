# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Building the Project

The project uses CMake for building. Essential build steps:

```bash
# Install dependencies first
sudo apt install cmake libjemalloc-dev libsnappy-dev libgflags-dev pkg-config swig libboost-all-dev libtbb-dev libisal-dev

# Compile isal-l_crypto
cd ThirdParty/isal-l_crypto
./autogen.sh
./configure
make -j

# Build SPFresh
mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug ..
make -j
```

### Key Executables

After building, these executables are available:
- `ssdserving` - Main SSD serving executable used in paper experiments
- `spfresh` - SPFresh main executable  
- `indexbuilder` - Build memory SPTAG indices
- `indexsearcher` - Search memory SPTAG indices
- `quantizer` - PQ quantizer training and vector quantization
- `server` - Vector search server
- `client` - Vector search client
- `usefultool` - Utility tool

### Python Package

The project can also be built as a Python package:

```bash
# Development installation
python setup.py develop

# Build wheel package
SPTAG_RELEASE=1.0 python setup.py bdist_wheel -p linux_x86_64
```

### Testing and Evaluation

For reproducing research experiments, use scripts in `Script_AE/`:
- Scripts correspond to figures in the paper (Figure1/, Figure6/, etc.)
- Requires Azure Standard_L16s_v3 instances for full reproduction
- See `Script_AE/README.md` for detailed instructions

```bash
# Bind NVMe device to SPDK
sudo nvme format /dev/nvme0n1
sudo ./ThirdParty/spdk/scripts/setup.sh
cp bdev.json ./

## Architecture Overview

### Core Components

**SPFresh** - The main system implementing LIRE (Live In-place REassignment) protocol for dynamic vector search with in-place updates.

**SPANN Architecture** - Built on SPANN (Space Partition and Approximate Nearest Neighbor) framework:
- `AnnService/inc/Core/SPANN/Index.h` - Main SPANN index implementation
- `AnnService/inc/Core/SPANN/ExtraRocksDBController.h` - RocksDB storage backend
- `AnnService/inc/Core/SPANN/ExtraSPDKController.h` - SPDK storage backend

**Storage Layers**:
- Memory layer: BKT (Ball k-d Tree) or KDT (k-d Tree) indices for head vectors
- Disk layer: RocksDB or SPDK-based storage for posting lists
- Two storage options: `ExtraRocksDBController` and `ExtraSPDKController`

**Key Algorithmic Components**:
- `AnnService/inc/Core/Common/RelativeNeighborhoodGraph.h` - Graph-based indexing
- `AnnService/inc/Core/Common/PQQuantizer.h` - Product Quantization for compression
- `AnnService/inc/SSDServing/SelectHead.h` - Head vector selection algorithms

### Build System

- CMake-based build system with complex third-party dependencies
- Options: `GPU` (GPU support), `ROCKSDB` (RocksDB backend), `LIBRARYONLY` (library-only build)
- Builds both shared (`SPTAGLib`) and static (`SPTAGLibStatic`) libraries
- Special handling for SIMD optimizations in `DistanceUtils`

### Data Flow

1. **Index Building**: Use `indexbuilder` with configuration files (`.ini` format)
2. **Head Selection**: Algorithms select representative vectors for memory storage
3. **Disk Index**: Posting lists stored on disk via RocksDB or SPDK
4. **Search Process**: Memory search followed by disk posting list retrieval
5. **Updates**: SPFresh implements in-place updates via LIRE protocol

### Configuration

The system uses extensive `.ini` configuration files with sections:
- `[Base]` - Basic parameters (dimensions, data types, file paths)
- `[SelectHead]` - Head selection parameters
- `[BuildHead]` - Head index building parameters  
- `[BuildSSDIndex]` - Disk index building parameters
- `[SearchSSDIndex]` - Search parameters

### File Formats

- **Vectors**: Binary format with header (num_vectors, dimensions) + raw data
- **Truth files**: Binary format (num_queries, K, truth_neighbor_ids)
- **Quantizer files**: Complex binary format for PQ/OPQ quantizers
- **Metadata**: Binary format with separate index file

## Important Notes

- Requires GCC 9+ for compilation due to C++17 features
- Memory-intensive - designed for high-memory Azure L-series instances  
- Multi-threaded throughout - thread counts configurable via parameters
- Extensive use of templates for different vector value types (Float, UInt8, Int16, etc.)
