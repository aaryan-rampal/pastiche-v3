# Pastiche: Sketch-to-Artwork Matching System

_Transform your hand-drawn sketches into connections with masterpiece artworks through advanced computer vision and shape analysis._

## 🎨 Project Overview

Pastiche is an intelligent web application that matches hand-drawn sketches to artwork from a curated collection using sophisticated contour analysis. Inspired by the concept of "land lines" connecting disparate locations through similar geographical features, Pastiche creates artistic connections by finding visual similarities between user sketches and historical artworks.

The system employs a two-stage matching pipeline combining FAISS (Facebook AI Similarity Search) for rapid initial filtering with Procrustes analysis for precise shape matching, enabling users to discover artworks that share visual DNA with their creative expressions.

## Why?

I have had this idea for a while, and have executed on two different occasions. Once at Hack the North 2023, another one after the hackathon since I didn't feel like I did a good enough job. Hence why this is Pastiche v3. So why again?

Because I can do better. I am better at ML, backend, frontend, and so is the technology. So why not give it one last shot?

## 🚀 Core Technology Stack

### Computer Vision Pipeline

- **OpenCV**: Contour extraction using Canny edge detection
- **Hu Moments**: 7-dimensional invariant shape descriptors for rotation/scale independence
- **FAISS**: Vector similarity search with ~400MB index for full dataset
- **Procrustes Analysis**: Precise shape alignment and similarity scoring
- **Scipy**: Statistical analysis and geometric transformations

### Backend Architecture

- **FastAPI**: High-performance Python web framework
- **Pydantic**: Data validation and serialization
- **AWS S3**: Cloud storage for 4,000+ artwork images
- **DynamoDB**: NoSQL database for artwork metadata
- **Boto3**: AWS SDK integration

### Frontend

- **React + TypeScript**: Modern component-based architecture
- **Canvas API**: Pressure-sensitive sketch drawing with smooth curves
- **Real-time Visualization**: Animated search effects and result display
- **Responsive Design**: Adaptive artwork presentation with aspect ratio preservation

## 🔬 Technical Deep Dive

### Shape Analysis Algorithm

1. **Contour Extraction**

   - Canny edge detection with adaptive thresholding
   - Contour filtering by area and perimeter
   - Noise reduction and shape simplification

2. **Feature Engineering**

   - Hu moments computation (7 invariant descriptors)
   - Enhanced feature vectors with area/perimeter ratios
   - Normalization for scale independence

[Check out the FAISS implementation notebook for a demo.](./data_analysis/eda/hu_faiss.ipynb)

3. **Two-Stage Matching**
   - **Stage 1**: FAISS approximate nearest neighbor search (fast, ~400ms)
   - **Stage 2**: Procrustes analysis on top candidates (precise, shape alignment)
   - **Scoring**: Combined similarity metric with confidence thresholds

### Performance Characteristics

- **Index Size**: 400MB for full dataset (4,000 artworks)
- **Search Speed**: <500ms for complete pipeline
- **Accuracy**: Procrustes scores of 0.06-0.07 for strong matches
- **Scalability**: Sub-linear search complexity with FAISS

## 📁 Project Structure

```
pastiche/
├── data/                           # Dataset and analysis
│   ├── classes_truncated.csv       # Artwork metadata (4K+ entries)
│   └── eda/                        # Exploratory data analysis
│       └── hu_faiss.py            # FAISS implementation
├── backend/                        # FastAPI application
│   ├── main.py                     # Application entry point
│   ├── core/                       # Configuration and settings
│   ├── routers/                    # API endpoints
│   ├── services/                   # Business logic
│   ├── models/                     # Pydantic data models
│   └── database/                   # DB connections
├── s3_setup/                       # Cloud infrastructure
│   ├── upload_to_s3.py            # Batch artwork upload
│   ├── setup_dynamodb.py          # Database initialization
│   └── setup_aws.sh               # Infrastructure automation
└── frontend/                       # React application (planned)
```

## 🎯 Key Features

### Intelligent Shape Matching

- **Rotation Invariant**: Matches regardless of sketch orientation
- **Scale Independent**: Works with sketches of any size
- **Noise Tolerant**: Handles imperfect hand-drawn lines
- **Multi-Contour**: Analyzes complex shapes with multiple components

### Curated Art Collection

- **4,000+ Artworks**: From classical to modern periods
- **Genre Diversity**: Baroque, Impressionism, Realism, Abstract, etc.
- **Rich Metadata**: Artist names, periods, descriptions
- **High Resolution**: Museum-quality images

### User Experience

- **Real-time Matching**: Instant results as you draw
- **Visual Feedback**: Confidence scores and match explanations
- **Educational Context**: Learn about discovered artworks
- **Discovery Mode**: Explore similar artworks and artistic movements

## 🔗 Inspiration: Land Lines

This project draws inspiration from "[Land Lines](https://lines.chromeexperiments.com/)" by Zach Lieberman, an interactive experiment that finds satellite imagery matching drawn lines. Pastiche extends this concept to the art world, creating connections between personal creativity and historical masterpieces through the universal language of shape and form.

Just as Land Lines reveals the hidden patterns in our planet's geography, Pastiche uncovers the visual relationships that connect human artistic expression across centuries and cultures.

## 🛠 Technical Implementation

### FAISS Index Configuration

```python
# Enhanced feature vector (9 dimensions)
features = np.array([
    hu_moments[0:7],           # 7 Hu moments
    area_ratio,                # Normalized area
    perimeter_ratio           # Normalized perimeter
])

# L2 distance with 100 nearest neighbors
index = faiss.IndexFlatL2(9)
index.add(features.astype('float32'))
```

### Procrustes Shape Analysis

```python
def procrustes_distance(shape1, shape2):
    # Align centroids
    shape1_centered = shape1 - shape1.mean(axis=0)
    shape2_centered = shape2 - shape2.mean(axis=0)

    # Scale normalization
    shape1_scaled = shape1_centered / np.sqrt((shape1_centered**2).sum())
    shape2_scaled = shape2_centered / np.sqrt((shape2_centered**2).sum())

    # Optimal rotation via SVD
    M = shape1_scaled.T @ shape2_scaled
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt

    # Compute alignment error
    aligned = shape2_scaled @ R.T
    return np.sqrt(((shape1_scaled - aligned)**2).sum())
```

## 📊 Dataset Statistics

- **Total Artworks**: 4,002 pieces
- **Artists**: 500+ including Van Gogh, Picasso, Monet, Rembrandt
- **Genres**: 15 major art movements
- **Time Period**: 15th century to modern era
- **Geographic Origin**: Global representation
- **Storage**: AWS S3 with organized folder structure

## 🚀 Getting Started

### Prerequisites

```bash
# Python environment
conda create -n pastiche python=3.12
conda activate pastiche

# Install dependencies
pip install -r requirements.txt

# AWS configuration
aws configure
```

### Development Setup

```bash
# Clone repository
git clone https://github.com/aaryan-rampal/pastiche-v3.git
cd pastiche-v3

# Backend development
cd backend
uvicorn main:app --reload --port 8000

# Frontend development (planned)
cd frontend
npm start
```

### AWS Infrastructure

```bash
cd s3_setup
chmod +x setup_aws.sh
./setup_aws.sh

python upload_to_s3.py      # Upload artwork collection
python setup_dynamodb.py    # Initialize metadata database
```

## 🔮 Future Enhancements

### Version 2.0 Roadmap

- **Style Transfer**: Generate artwork in the style of matched pieces
- **Color Analysis**: Incorporate color harmony in matching algorithm
- **Social Features**: Share discoveries and build sketch galleries

### Advanced Algorithms

- **Deep Learning**: CNN-based feature extraction for complex patterns
- **Graph Neural Networks**: Relationship modeling between artworks
- **Attention Mechanisms**: Focus on significant shape components
- **Multi-modal Fusion**: Combine shape, texture, and color features

## 🤝 Contributing

We welcome contributions from artists, developers, and art enthusiasts! Areas of focus:

- **Algorithm Improvements**: Enhanced matching accuracy
- **Dataset Expansion**: Additional artwork collections
- **UI/UX Design**: Intuitive user interfaces
- **Performance Optimization**: Faster search and matching
- **Documentation**: Code examples and tutorials

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Land Lines**: Original inspiration by Zach Lieberman
- **WikiArt**: Artwork dataset and metadata
- **OpenCV Community**: Computer vision algorithms
- **Facebook Research**: FAISS similarity search library

---

## 📋 Project Progress

### ✅ Completed Phases

#### Phase 1: Research & Analysis (Complete)

- [x] **Dataset Acquisition**: Curated 4,000+ artwork collection from WikiArt
- [x] **Exploratory Data Analysis**: Statistical analysis of artwork distribution
- [x] **Algorithm Research**: Evaluated shape matching approaches (Hu moments, Fourier descriptors, SIFT)
- [x] **Prototype Development**: Initial contour extraction and matching pipeline

#### Phase 2: Core Algorithm Implementation (Complete)

- [x] **Contour Extraction**: OpenCV-based edge detection with noise filtering
- [x] **Feature Engineering**: Hu moments computation with enhanced descriptors
- [x] **FAISS Integration**: Vector similarity search with optimized indexing
- [x] **Procrustes Analysis**: Precise shape alignment and scoring
- [x] **Performance Optimization**: Achieved <500ms end-to-end matching

#### Phase 3: Cloud Infrastructure (Complete)

- [x] **AWS Setup**: S3 bucket configuration with proper IAM permissions
- [x] **Data Upload**: Batch upload of 4,000+ artworks with metadata preservation
- [ ] **DynamoDB Design**: NoSQL schema for artwork metadata and search indexing
- [ ] **Infrastructure Automation**: Scripts for reproducible cloud deployment

### ✅ Current Phase: Full-Stack Implementation (Complete)

#### Phase 4: API Development (Complete)

- [x] **FastAPI Framework**: Project structure and configuration with S3 integration
- [x] **Data Models**: Pydantic schemas for request/response validation
- [x] **Core Endpoints**:
  - `/api/sketch/match-points`: Point-based sketch matching with exponential distribution selection
  - `/api/sketch/image/{path}`: S3 image proxy to avoid CORS issues
  - `/api/sketch/health`: Service health monitoring
- [x] **FAISS Service**: Singleton pattern with lazy loading and metadata management
- [x] **Procrustes Service**:
  - Parallel S3 fetching with ThreadPoolExecutor (10 concurrent workers)
  - Batch contour extraction and alignment
  - Full transformation parameters (scale, rotation, translation)
- [x] **Error Handling**: Comprehensive exception management with HTTP status codes
- [x] **Performance Optimization**:
  - Parallelized I/O operations
  - Efficient numpy operations
  - Smart caching with proxy endpoints

#### Phase 5: User Interface (Complete)

- [x] **React + TypeScript Setup**: Modern component-based architecture with Vite
- [x] **Canvas Interface**:
  - Pressure-sensitive drawing with smooth quadratic curves
  - FIFO queue for point management (max 75 points)
  - Real-time stroke rendering with variable line width
- [x] **Result Display**:
  - Immersive full-screen canvas experience
  - Animated search states with pulsing effects and random ripple waves
  - Dynamic artwork presentation with contour highlighting
- [x] **Responsive Design**:
  - Dark/light mode support with localStorage persistence
  - Adaptive image sizing (40-70% of viewport with random positioning)
  - Aspect ratio preservation with automatic scaling
- [x] **Advanced Features**:
  - **MatchedArtwork Component**: Self-contained image display with:
    - Automatic aspect ratio preservation
    - Intelligent random positioning (centered with ±15% variance)
    - Smart contour overlay scaling (natural → rendered dimensions)
    - Canvas-based contour highlighting with glow effects
  - **Visual Feedback**: Color-coded contours (green in dark mode, pink in light)
  - **Error States**: User-friendly error messages with fallback handling

### 🚀 Future Phases

#### Phase 6: Production Deployment (Planned)
- [ ] **CI/CD Pipeline**: Automated testing and deployment
- [ ] **Monitoring**: Application performance and error tracking
- [ ] **Scaling**: Load balancing and database optimization
- [ ] **Security**: HTTPS, rate limiting, and input validation
- [ ] **Domain Setup**: Custom domain with SSL certificates

#### Phase 7: Enhancement & Growth (Planned)
- [ ] **User Analytics**: Engagement tracking and behavior analysis
- [ ] **A/B Testing**: Algorithm and interface optimization
- [ ] **Community Features**: User galleries and social sharing
- [ ] **Mobile Apps**: Native iOS and Android applications
- [ ] **Advanced Matching**: Multi-contour sketches and color matching
- [ ] **Educational Features**: Artwork information and historical context

---

## 🎯 Key Technical Achievements

### Backend Innovations

1. **Hybrid Matching Algorithm**:
   - FAISS retrieves top 1000 candidates in ~100ms
   - Procrustes refines to top 10 matches
   - Exponential distribution for intelligent random selection (favors best matches)

2. **Performance Optimization**:
   - Parallelized S3 fetching (10x speedup with ThreadPoolExecutor)
   - Smart image proxy to eliminate CORS issues
   - Singleton pattern for FAISS index (loaded once, reused across requests)

3. **Robust Error Handling**:
   - Graceful degradation with detailed error messages
   - Per-candidate error isolation in batch processing
   - Comprehensive logging with loguru

### Frontend Innovations

1. **Advanced Canvas Drawing**:
   - Smooth quadratic curves using control points
   - Variable line width based on simulated pressure
   - FIFO queue prevents memory issues with long strokes
   - Shadow effects for visual depth

2. **Intelligent Result Display**:
   - Aspect-ratio-aware image rendering
   - Dynamic contour overlay with coordinate transformation
   - Scale factor calculation: `scaleX = renderedWidth / naturalWidth`
   - Random yet aesthetic positioning algorithm

3. **Visual Design**:
   - Immersive full-screen experience
   - Animated search states with 11 random messages
   - Distributed glow effects across stroke points
   - Random ripple waves for ambient animation
   - Theme-aware color schemes (dark/light mode)

---

_Last Updated: September 2025 | Current Status: Full-Stack MVP Complete | Next Milestone: Production Deployment_
