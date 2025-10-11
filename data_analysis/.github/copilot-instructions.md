# GitHub Copilot Instructions for Data Analysis

## Project Overview
Pastiche data_analysis contains exploratory data analysis (EDA), machine learning model definitions, and utility functions for the sketch-to-artwork matching system. This component handles data preprocessing, feature engineering, FAISS index building, and algorithm development for the computer vision pipeline.

## Core Technologies
- **NumPy** for numerical computations and array operations
- **OpenCV** for computer vision and image processing
- **Pandas** for data manipulation and CSV processing
- **Matplotlib** for visualization and plotting
- **Scikit-learn** for machine learning utilities
- **FAISS** for efficient similarity search and indexing
- **Jupyter Notebook** for interactive analysis and prototyping
- **Pydantic** for data validation and model definitions

## Key Components

### Models (data_analysis/eda/models.py)
- **Contour**: Pydantic model for individual contour data with validation
- **ImageModel**: Container for image metadata and associated contours
- **ProcrustesResult**: Complete transformation results from shape alignment
- **ContourFAISSIndex**: FAISS index wrapper with metadata management

### Utils (data_analysis/eda/utils.py)
- **compute_hu_moments()**: Hu invariant moments calculation for shape descriptors
- **compute_enhanced_features()**: Extended feature vectors with shape metrics
- **Image Processing**: Contour extraction and feature computation utilities

### Main Analysis Scripts
- **hu_faiss.py**: Core FAISS index building and two-stage matching pipeline
- **contours.py**: Contour extraction and preprocessing utilities
- **models.py**: Data models and validation schemas

## Data Pipeline

### Data Sources
- **Artwork Metadata**: CSV files with artist, period, and image information
- **Image Collections**: Local directories or S3 buckets with artwork images
- **Sketch Data**: User-drawn contours for testing and validation

### Processing Steps
1. **Data Loading**: Read CSV metadata and validate image paths
2. **Image Processing**: Load images and extract contours using OpenCV
3. **Feature Engineering**: Compute Hu moments and enhanced descriptors
4. **Index Building**: Create FAISS index for efficient similarity search
5. **Validation**: Test matching pipeline with sample sketches

## Feature Engineering

### Hu Moments (7 dimensions)
- **Invariant Descriptors**: Rotation, scale, and translation invariant
- **Numerical Stability**: Log-scale transformation for better conditioning
- **Shape Representation**: Captures complex shape characteristics

### Enhanced Features (15 dimensions)
- **Hu Moments**: Base 7 invariant shape descriptors
- **Compactness**: Circularity measure (4π × area / perimeter²)
- **Aspect Ratio**: Width-to-height ratio from bounding rectangle
- **Extent**: Contour area relative to bounding rectangle
- **Solidity**: Contour area relative to convex hull
- **Normalized Length**: Contour point count relative to image size

## FAISS Index Management

### Index Configuration
- **Algorithm**: L2 distance (Euclidean) for similarity search
- **Dimensions**: 7 (Hu moments) or 15 (enhanced features)
- **Normalization**: Feature scaling for improved performance

### Metadata Management
- **Contour Mapping**: Index positions to (image_path, contour_idx) pairs
- **S3 Compatibility**: Separate metadata for cloud storage paths
- **Persistence**: Save/load index and metadata to disk

## Two-Stage Matching Algorithm

### Stage 1: FAISS Search
- **Input**: Sketch contour points
- **Process**: Compute Hu moments, search FAISS index
- **Output**: Top-k candidate contours with similarity scores
- **Performance**: ~100ms for 40,000+ contour search

### Stage 2: Procrustes Refinement
- **Input**: Top candidates from FAISS
- **Process**: Precise shape alignment using Procrustes analysis
- **Output**: Transformation parameters and disparity scores
- **Performance**: Detailed alignment for top matches

## Procrustes Analysis

### Shape Alignment Process
1. **Centroid Translation**: Move shapes to common center
2. **Scale Normalization**: Equalize shape sizes
3. **Rotation Optimization**: SVD-based optimal rotation finding
4. **Disparity Calculation**: Measure alignment quality

### Transformation Parameters
- **Scale Factor**: Size adjustment for sketch-to-target matching
- **Rotation Matrix**: 2D rotation from SVD decomposition
- **Translation Vector**: Position offset between shapes
- **Centroid Coordinates**: Reference points for alignment

## Data Visualization

### Contour Visualization
- **Original Images**: Display source artwork with extracted contours
- **Overlay Plots**: Show sketch and target contours together
- **Transformation Display**: Visualize alignment results
- **Comparison Metrics**: Plot similarity scores and transformation parameters

### Analysis Plots
- **Feature Distributions**: Histograms of Hu moments and shape metrics
- **Similarity Matrices**: Heatmaps of contour-to-contour distances
- **Performance Metrics**: Accuracy plots and timing analysis

## Development Workflow

### Exploratory Analysis
1. **Data Exploration**: Load and examine artwork metadata
2. **Image Processing**: Extract contours from sample images
3. **Feature Analysis**: Compute and visualize shape descriptors
4. **Index Building**: Create FAISS index for testing

### Algorithm Development
1. **Prototype**: Implement new matching algorithms in Jupyter
2. **Testing**: Validate with sample sketches and known matches
3. **Optimization**: Profile and improve performance bottlenecks
4. **Integration**: Move validated code to production services

## Code Conventions

### Import Organization
- **Standard Library**: numpy, cv2, matplotlib, etc.
- **Local Imports**: Relative imports for project modules
- **Type Hints**: Full type annotations for function signatures

### Data Structures
- **NumPy Arrays**: Primary data structure for contours and features
- **Pydantic Models**: Validation and serialization for complex objects
- **Dictionaries**: Metadata storage and configuration

### Error Handling
- **Graceful Degradation**: Skip invalid contours during processing
- **Logging**: Progress tracking for long-running operations
- **Validation**: Input checking with informative error messages

## Common Tasks

### Building FAISS Index
1. Load artwork metadata from CSV
2. Process images in batches with progress tracking
3. Extract and validate contours for each image
4. Compute features and build FAISS index
5. Save index and metadata for backend use

### Testing Matching Pipeline
1. Load pre-built FAISS index
2. Prepare test sketch contours
3. Run two-stage matching algorithm
4. Visualize results and transformation parameters
5. Evaluate accuracy and performance metrics

### Feature Engineering
1. Analyze existing feature distributions
2. Implement new shape descriptors
3. Test feature combinations for matching accuracy
4. Update FAISS index with enhanced features

## Performance Optimization

### Processing Speed
- **Batch Processing**: Handle multiple images simultaneously
- **Parallel Computation**: Use ProcessPoolExecutor for CPU-intensive tasks
- **Memory Efficiency**: Stream processing for large datasets

### Index Optimization
- **Feature Normalization**: Improve FAISS search quality
- **Index Parameters**: Tune for specific dataset characteristics
- **Metadata Compression**: Efficient storage of contour mappings

## Integration Points

### Backend Services
- **Shared Models**: Common Pydantic schemas across components
- **FAISS Index**: Built here, loaded by FAISSService
- **Feature Consistency**: Same computation logic in utils

### Frontend Integration
- **Data Formats**: Compatible contour and transformation data
- **Visualization**: Results formatted for frontend display
- **API Compatibility**: Match backend endpoint expectations

## Debugging and Validation

### Contour Extraction
- **Visual Inspection**: Plot extracted contours on original images
- **Parameter Tuning**: Adjust Canny thresholds and filtering
- **Quality Metrics**: Validate contour area and complexity

### Feature Computation
- **Numerical Stability**: Check for NaN/inf values in Hu moments
- **Distribution Analysis**: Plot feature histograms for anomalies
- **Correlation Analysis**: Examine relationships between features

### Matching Validation
- **Ground Truth**: Use known similar artworks for testing
- **Visual Verification**: Manually check alignment quality
- **Metric Analysis**: Evaluate precision and recall of matches

## Research and Experimentation

### Algorithm Variations
- **Alternative Features**: SIFT, SURF, or deep learning descriptors
- **Distance Metrics**: Cosine, Manhattan, or learned metrics
- **Matching Strategies**: Different two-stage or multi-stage approaches

### Data Augmentation
- **Synthetic Sketches**: Generate training data programmatically
- **Transformation Invariance**: Test rotation, scale, translation robustness
- **Noise Tolerance**: Evaluate performance with imperfect inputs

## File Organization

### Directory Structure
```
data_analysis/
├── eda/                    # Exploratory data analysis
│   ├── models.py          # Data models and validation
│   ├── utils.py           # Feature computation utilities
│   ├── hu_faiss.py        # Main FAISS pipeline
│   ├── contours.py        # Contour processing
│   └── hu_faiss.ipynb     # Jupyter notebook version
├── sketches/              # Test sketch data
└── drawing_app.py         # Drawing application for testing
```

### Data Dependencies
- **Input Data**: CSV metadata and image collections
- **Output Artifacts**: FAISS index files and metadata pickles
- **Intermediate Results**: Contour extractions and feature vectors

## Future Enhancements

### Advanced Features
- **Deep Learning**: CNN-based feature extraction
- **Multi-modal Matching**: Combine shape, color, texture
- **Temporal Analysis**: Match sequences of sketches
- **Interactive Learning**: User feedback for model improvement

### Scalability Improvements
- **Distributed Processing**: Handle larger datasets with Dask/Spark
- **GPU Acceleration**: CUDA-accelerated FAISS and computer vision
- **Streaming Pipeline**: Real-time index updates and matching

### Research Directions
- **Shape Understanding**: Semantic interpretation of contours
- **Style Transfer**: Generate artwork in matched style
- **Cultural Analysis**: Discover artistic influences and connections
