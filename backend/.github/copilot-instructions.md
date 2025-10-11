# GitHub Copilot Instructions for Backend

## Project Overview
Pastiche backend is a FastAPI application that provides sketch-to-artwork matching services using advanced computer vision techniques. The system implements a two-stage matching pipeline combining FAISS for rapid initial filtering with Procrustes analysis for precise shape matching.

## Core Technologies
- **FastAPI** for high-performance Python web framework
- **Pydantic** for data validation and serialization
- **OpenCV** for contour extraction and image processing
- **FAISS** (Facebook AI Similarity Search) for vector similarity search
- **AWS SDK (Boto3)** for S3 and DynamoDB integration
- **NumPy** for numerical computations and array operations
- **SciPy** for statistical analysis and geometric transformations
- **Loguru** for structured logging

## Key Services

### FAISSService (backend/services/faiss_service.py)
- **Purpose**: Singleton service for FAISS index management and similarity search
- **Features**:
  - Lazy loading of FAISS index and metadata
  - Enhanced feature vectors using Hu moments (7 dimensions)
  - L2 distance similarity search with configurable top-k
  - Metadata management for contour-to-artwork mapping
- **Methods**: `search_similar_contours()`, `get_metadata()`, `_load_index()`

### ProcrustesService (backend/services/procrustes_service.py)
- **Purpose**: Precise shape matching using Procrustes analysis
- **Features**:
  - Parallel S3 fetching with ThreadPoolExecutor (10 concurrent workers)
  - Batch contour extraction and alignment
  - Full transformation parameters (scale, rotation, translation)
  - Exponential distribution for intelligent random selection
- **Methods**: `compute_procrustes_batch()`, `_fetch_contour_from_s3()`, `align_contours()`

### ContourService (backend/services/contour_service.py)
- **Purpose**: Image processing and contour extraction
- **Features**:
  - Canny edge detection with adaptive thresholding
  - Contour filtering by area and perimeter
  - Noise reduction and shape simplification
  - Hu moments computation for invariant shape descriptors
- **Methods**: `extract_contours_from_image_bytes()`, `compute_hu_moments()`

## API Endpoints

### POST /api/sketch/match-points
- **Purpose**: Match sketch points to artwork database
- **Request**: PointInput with sketch contour points and search parameters
- **Response**: MatchResponse with matched artworks and transformation data
- **Flow**: Points → FAISS search → Procrustes refinement → Exponential selection

### GET /api/sketch/image/{path:path}
- **Purpose**: Proxy endpoint for artwork images from S3
- **Parameters**: S3 path (e.g., 'artworks/baroque/image.jpg')
- **Response**: StreamingResponse with image bytes
- **Features**: CORS bypass, caching headers, content-type detection

### GET /api/sketch/health
- **Purpose**: Health check endpoint
- **Response**: Status and FAISS index information
- **Features**: Index loading verification, contour count reporting

## Data Models

### PointInput (backend/models/schemas.py)
- **Fields**: `points: List[List[float]]` - Sketch contour points
- **Validation**: Minimum 3 points required for matching

### MatchResult (backend/models/schemas.py)
- **Fields**: artwork_path, procrustes_score, hu_distance, transform, contour_idx, matched_contour_points
- **Features**: Complete transformation parameters for frontend positioning

### TransformParams (backend/models/schemas.py)
- **Fields**: scale, rotation_degrees, rotation_radians, translation, sketch_centroid, target_centroid
- **Purpose**: Full transformation data for sketch-to-artwork alignment

## Architecture Patterns

### Singleton Pattern
- **FAISSService**: Single instance with lazy-loaded index
- **Benefits**: Memory efficiency, consistent state, shared resources

### Service Layer Architecture
- **Separation of Concerns**: Services handle business logic, routers handle HTTP
- **Dependency Injection**: Services instantiated at module level
- **Error Handling**: Comprehensive exception management with HTTP status codes

### Two-Stage Matching Pipeline
1. **Stage 1**: FAISS approximate nearest neighbor search (~100ms)
2. **Stage 2**: Procrustes analysis on top candidates (precise alignment)
3. **Selection**: Exponential distribution favors better matches

## Performance Optimization

### Parallel Processing
- **S3 Fetching**: ThreadPoolExecutor with 10 concurrent workers
- **Procrustes Computation**: Batch processing with progress tracking
- **I/O Operations**: Non-blocking image retrieval from S3

### Memory Management
- **FAISS Index**: Loaded once and reused across requests
- **Contour Processing**: Efficient numpy operations
- **Streaming Responses**: Large images served as streams

### Caching Strategies
- **Image Proxy**: Avoids CORS issues and enables caching
- **FAISS Index**: In-memory singleton with lazy loading
- **Transform Parameters**: Computed on-demand for each match

## AWS Integration

### S3 Configuration
- **Bucket**: Artwork storage with organized folder structure
- **Access**: Boto3 client with proper IAM permissions
- **Proxy**: Custom endpoint for image serving

### DynamoDB (Planned)
- **Purpose**: NoSQL database for artwork metadata
- **Schema**: Artist names, periods, descriptions, search indices
- **Integration**: Future enhancement for metadata management

## Configuration

### Settings (backend/core/config.py)
- **Environment Variables**: CORS origins, debug mode, FAISS parameters
- **Default Values**: Sensible defaults for development
- **Type Safety**: Pydantic BaseSettings for validation

### Development Setup
- **Local Testing**: uvicorn with auto-reload
- **Environment**: Separate configs for development/production
- **Logging**: Structured logging with loguru

## Error Handling

### HTTP Exception Management
- **Validation Errors**: 400 status codes for bad requests
- **Resource Not Found**: 404 for missing images
- **Internal Errors**: 500 for processing failures

### Service-Level Error Handling
- **FAISS Errors**: Graceful fallback when index not loaded
- **S3 Errors**: Timeout handling and retry logic
- **Contour Processing**: Validation and filtering of invalid contours

## Development Workflow

### Local Development
1. **Setup**: `pip install -r requirements.txt`
2. **Run**: `uvicorn main:app --reload --port 8000`
3. **Test**: Use `/docs` for interactive API documentation

### Testing
- **Unit Tests**: Service layer testing with mock data
- **Integration Tests**: End-to-end API testing
- **Performance Tests**: Load testing for matching endpoints

## Code Conventions

### Import Organization
- **Standard Library**: First block
- **Third Party**: Second block
- **Local Imports**: Third block with relative imports

### Type Hints
- **Function Parameters**: Full type annotations
- **Return Types**: Explicit return type hints
- **Complex Types**: Use `typing` module for generics

### Error Handling
- **Specific Exceptions**: Catch specific exception types
- **Logging**: Use loguru for structured logging
- **HTTP Responses**: Use FastAPI's HTTPException

## Common Tasks

### Adding New Matching Algorithm
1. Create new service class in `services/`
2. Implement algorithm with proper error handling
3. Add configuration parameters to `core/config.py`
4. Update router to expose new endpoint
5. Add Pydantic models for request/response

### Optimizing Performance
1. Profile current bottlenecks using Python profilers
2. Implement parallel processing where applicable
3. Add caching layers for expensive operations
4. Optimize numpy operations for vectorization

### Adding New Data Sources
1. Extend S3 integration for new bucket structures
2. Update metadata models for additional fields
3. Modify FAISS index building for new data
4. Add validation for new data formats

## Integration Points

### Frontend Communication
- **CORS**: Configured for localhost development
- **Data Format**: JSON with numpy array serialization
- **Error Responses**: Structured error messages

### Data Analysis Pipeline
- **Shared Models**: Common Pydantic models across components
- **FAISS Index**: Built by data_analysis, loaded by backend
- **Feature Computation**: Consistent Hu moments calculation

### AWS Infrastructure
- **S3 Access**: Boto3 client with environment-based credentials
- **Image Serving**: Proxy endpoints for frontend consumption
- **Metadata Storage**: Future DynamoDB integration

## Debugging Tips

### API Debugging
- Use `/docs` for interactive endpoint testing
- Check logs for detailed error information
- Monitor FAISS index loading status

### Performance Debugging
- Profile with `cProfile` for bottleneck identification
- Monitor memory usage with `memory_profiler`
- Use logging to track processing times

### Computer Vision Debugging
- Visualize contours with OpenCV drawing functions
- Check Hu moments computation for numerical stability
- Validate Procrustes transformation parameters

## Security Considerations

### Input Validation
- **Point Limits**: Maximum points per sketch request
- **Image Size**: Reasonable limits on processed images
- **Path Traversal**: Sanitize S3 paths in proxy endpoints

### AWS Security
- **IAM Permissions**: Least privilege for S3 access
- **Environment Variables**: Secure credential management
- **Network Security**: VPC configuration for production

## Future Enhancements

### Planned Features
- **DynamoDB Integration**: Metadata storage and querying
- **Batch Processing**: Multiple sketches in single request
- **Advanced Matching**: Color and texture analysis
- **Real-time Updates**: WebSocket support for progress

### Scalability Improvements
- **Load Balancing**: Multiple backend instances
- **Caching Layer**: Redis for frequently accessed data
- **Database Indexing**: Optimized metadata queries
