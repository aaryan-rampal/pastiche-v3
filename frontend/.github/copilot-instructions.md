# GitHub Copilot Instructions for Frontend

## Project Overview
Pastiche frontend is a React + TypeScript + Vite application that provides an immersive canvas drawing interface for matching hand-drawn sketches to artwork. The application features real-time drawing, animated search states, and responsive artwork display with contour highlighting.

## Core Technologies
- **React 18** with TypeScript for component-based architecture
- **Vite** for fast development and building
- **Tailwind CSS** for styling with dark/light mode support
- **Canvas API** for pressure-sensitive drawing and smooth curves
- **ESLint** for code quality

## Key Components

### ImmersiveCanvas (src/components/ImmersiveCanvas.tsx)
- **Purpose**: Main drawing canvas and search interface
- **Features**:
  - Pressure-sensitive drawing with smooth quadratic curves
  - FIFO queue for point management (max 75 points)
  - Real-time stroke rendering with variable line width
  - Animated search states with pulsing effects and random ripple waves
  - Dynamic artwork presentation with contour highlighting
- **Props**: isDarkMode, onStrokeCountChange, onSearchTrigger, onClear, maxStrokes, isSearching

### FloatingControls (src/components/FloatingControls.tsx)
- **Purpose**: UI controls for theme switching and canvas management
- **Features**: Dark/light mode toggle, clear canvas button
- **Props**: isDarkMode, onToggleDarkMode

### MatchedArtwork (src/components/MatchedArtwork.tsx)
- **Purpose**: Display matched artwork with contour overlay
- **Features**:
  - Automatic aspect ratio preservation
  - Intelligent random positioning (centered with ±15% variance)
  - Smart contour overlay scaling (natural → rendered dimensions)
  - Canvas-based contour highlighting with glow effects
- **Props**: artworkUrl, contourPoints, isDarkMode

## State Management
- **Dark/Light Mode**: Persistent localStorage preference
- **Stroke Count**: Tracks drawing progress (max 25 strokes)
- **Search State**: Manages animated search states
- **Canvas State**: Points queue and rendering context

## Canvas Drawing Features
- **Smooth Curves**: Quadratic Bezier curves with control points
- **Pressure Simulation**: Variable line width based on drawing speed
- **Point Management**: FIFO queue prevents memory issues
- **Shadow Effects**: Visual depth for strokes

## API Integration
- **Base URL**: Backend API (localhost:8000 in development)
- **Endpoints**:
  - `POST /api/sketch/match-points`: Submit sketch points for matching
  - `GET /api/sketch/image/{path}`: Proxy for artwork images
- **Data Flow**: Sketch points → FAISS search → Procrustes refinement → Match display

## Visual Design
- **Immersive Experience**: Full-screen canvas with minimal UI
- **Theme System**: Dark mode (green contours) / Light mode (pink contours)
- **Animations**: Search states with 11 random messages, ripple waves
- **Responsive Layout**: Adaptive sizing (40-70% viewport with random positioning)

## Development Workflow
1. **Development**: `npm run dev` (Vite dev server)
2. **Build**: `npm run build` (production bundle)
3. **Lint**: `npm run lint` (ESLint checking)
4. **Type Check**: `npm run type-check` (TypeScript validation)

## Code Conventions
- **TypeScript**: Strict typing with interfaces for props and state
- **Components**: Functional components with hooks
- **Styling**: Tailwind CSS utility classes
- **File Structure**: Component files in src/components/
- **Asset Management**: Static assets in src/assets/

## Performance Considerations
- **Canvas Optimization**: Efficient point management and rendering
- **Image Loading**: Lazy loading for artwork images
- **Animation Throttling**: RequestAnimationFrame for smooth animations
- **Memory Management**: FIFO queue prevents memory leaks

## Common Tasks

### Adding New Drawing Features
1. Extend ImmersiveCanvas component with new drawing modes
2. Update point processing logic in canvas event handlers
3. Add corresponding UI controls in FloatingControls

### Modifying Search Animation
1. Update search messages in ImmersiveCanvas
2. Modify animation timing and effects
3. Ensure theme-aware color schemes

### Adding New Display Components
1. Create new component in src/components/
2. Implement TypeScript interfaces for props
3. Use Tailwind CSS for styling
4. Integrate with App.tsx state management

## Error Handling
- **Canvas Errors**: Graceful fallback for canvas initialization failures
- **API Errors**: User-friendly error messages for network issues
- **Loading States**: Visual feedback during search operations

## Integration Points
- **Backend Communication**: Fetch API for sketch matching
- **Theme System**: CSS classes and localStorage persistence
- **Canvas Rendering**: Direct DOM manipulation for drawing
- **Image Display**: Proxy endpoints to avoid CORS issues

## Debugging Tips
- Use browser DevTools for canvas rendering inspection
- Check network tab for API request/response debugging
- Use React DevTools for component state inspection
- Monitor console for drawing performance metrics
