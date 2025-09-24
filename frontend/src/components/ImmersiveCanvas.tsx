import React, { useRef, useEffect, useState, useCallback } from 'react'

interface StrokePoint {
  x: number
  y: number
  pressure: number
}

interface StrokePath {
  points: StrokePoint[]
  id: number
  isComplete: boolean
}

interface ImmersiveCanvasProps {
  isDarkMode: boolean
}

const ImmersiveCanvas: React.FC<ImmersiveCanvasProps> = ({ isDarkMode }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const contextRef = useRef<CanvasRenderingContext2D | null>(null)
  const [strokePaths, setStrokePaths] = useState<StrokePath[]>([])
  const [currentPath, setCurrentPath] = useState<StrokePoint[]>([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [isAnimating, setIsAnimating] = useState(false)
  const [currentMessage, setCurrentMessage] = useState('')

  const maxStrokes = 25
  const maxContourLength = 25 // Maximum pixels for a single contour
  const strokeIdRef = useRef(0)
  const animationTimeoutRef = useRef<number | null>(null)

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const context = canvas.getContext('2d')
    if (!context) return

    // Set canvas size to full window
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    contextRef.current = context

    return () => window.removeEventListener('resize', resizeCanvas)
  }, [])

  // Helper function to calculate path length
  const calculatePathLength = (points: StrokePoint[]): number => {
    if (points.length < 2) return 0

    let totalLength = 0
    for (let i = 1; i < points.length; i++) {
      const dx = points[i].x - points[i - 1].x
      const dy = points[i].y - points[i - 1].y
      totalLength += Math.sqrt(dx * dx + dy * dy)
    }
    return totalLength
  }

  // Helper function to complete current stroke
  const completeCurrentStroke = useCallback(() => {
    if (currentPath.length === 0) return

    const newPath: StrokePath = {
      points: currentPath,
      id: strokeIdRef.current++,
      isComplete: true
    }

    setStrokePaths(prev => {
      const updated = [...prev, newPath]
      // Keep only the latest 25 stroke paths (instant removal)
      return updated.slice(-maxStrokes)
    })

    // Clear current path to start a new one
    setCurrentPath([])
  }, [currentPath])

  // Render smooth stroke paths
  useEffect(() => {
    const canvas = canvasRef.current
    const context = contextRef.current
    if (!canvas || !context) return

    // Clear canvas
    context.clearRect(0, 0, canvas.width, canvas.height)

    // Set stroke color based on theme
    const strokeColor = isDarkMode ? '#f5f2ed' : '#1a1614'

    // Draw completed stroke paths
    strokePaths.forEach((strokePath) => {
      if (strokePath.points.length < 2) return

      context.save()
      context.strokeStyle = strokeColor
      context.lineCap = 'round'
      context.lineJoin = 'round'
      context.shadowColor = strokeColor
      context.shadowBlur = 2

      // Draw smooth path using quadratic curves
      context.beginPath()
      context.moveTo(strokePath.points[0].x, strokePath.points[0].y)

      const start = strokePath.points.length - 25 > 0 ? strokePath.points.length - 25 : 0
      for (let i = start; i < strokePath.points.length - 1; i++) {
        const currentPoint = strokePath.points[i]
        const nextPoint = strokePath.points[i + 1]

        // Variable line width based on pressure
        const pressure = currentPoint.pressure
        context.lineWidth = 3 + pressure * 8

        // Use quadratic curve for smoothness
        const controlX = (currentPoint.x + nextPoint.x) / 2
        const controlY = (currentPoint.y + nextPoint.y) / 2
        context.quadraticCurveTo(currentPoint.x, currentPoint.y, controlX, controlY)
      }

      // Handle last point
      if (strokePath.points.length > 1) {
        const lastPoint = strokePath.points[strokePath.points.length - 1]
        context.lineTo(lastPoint.x, lastPoint.y)
      }

      context.stroke()
      context.restore()
    })

    // Draw current path being drawn
    if (currentPath.length > 1) {
      context.save()
      context.strokeStyle = strokeColor
      context.lineCap = 'round'
      context.lineJoin = 'round'
      context.shadowColor = strokeColor
      context.shadowBlur = 2

      context.beginPath()
      context.moveTo(currentPath[0].x, currentPath[0].y)

      for (let i = 1; i < currentPath.length - 1; i++) {
        const currentPoint = currentPath[i]
        const nextPoint = currentPath[i + 1]

        const pressure = currentPoint.pressure
        context.lineWidth = 3 + pressure * 8

        const controlX = (currentPoint.x + nextPoint.x) / 2
        const controlY = (currentPoint.y + nextPoint.y) / 2
        context.quadraticCurveTo(currentPoint.x, currentPoint.y, controlX, controlY)
      }

      if (currentPath.length > 1) {
        const lastPoint = currentPath[currentPath.length - 1]
        context.lineTo(lastPoint.x, lastPoint.y)
      }

      context.stroke()
      context.restore()
    }
  }, [strokePaths, currentPath, isDarkMode])

  // Start drawing
  const startDrawing = useCallback((event: React.MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    // Clear all existing stroke paths when starting new drawing session
    setStrokePaths([])

    // Start new path
    setCurrentPath([{ x, y, pressure: 0.8 }])
    setIsDrawing(true)
    setIsAnimating(false)

    if (animationTimeoutRef.current) {
      clearTimeout(animationTimeoutRef.current)
    }
  }, [])

  // Draw stroke
  const draw = useCallback((event: React.MouseEvent) => {
    if (!isDrawing) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    // Calculate pressure from movement speed (simulated)
    const pressure = Math.min(1, Math.max(0.3, Math.random() * 0.4 + 0.6))

    // Create new point
    const newPoint: StrokePoint = { x, y, pressure }

    // Check if adding this point would exceed the maximum contour length
    const potentialPath = [...currentPath, newPoint]
    const pathLength = calculatePathLength(potentialPath)

    if (pathLength > maxContourLength && currentPath.length > 1) {
      // Complete current stroke and start a new one
      completeCurrentStroke()
      // Start new path with this point
      setCurrentPath([newPoint])
    } else {
      // Add point to current path
      setCurrentPath(prev => [...prev, newPoint])
    }
  }, [isDrawing, currentPath, completeCurrentStroke])

  // Stop drawing and trigger animation
  const stopDrawing = useCallback(() => {
    if (!isDrawing) return

    setIsDrawing(false)

    // Complete the current path if it has points
    completeCurrentStroke()

    const messages = [
      "Finding the perfect artwork...",
      "Discovering masterpiece matches...",
      "Connecting with art history...",
      "Searching the masters...",
      "Exploring artistic connections...",
      "Unveiling hidden similarities..."
    ]

    // Trigger animation immediately when user stops drawing
    if (currentPath.length > 0) {
      setIsAnimating(true)
      setCurrentMessage(messages[Math.floor(Math.random() * messages.length)])

      // Stop animation after 4 seconds
      setTimeout(() => {
        setIsAnimating(false)
        setCurrentMessage('')
      }, 4000)
    }
  }, [isDrawing, currentPath, completeCurrentStroke])  // Render glow effects during animation
  const renderGlowEffects = () => {
    if (!isAnimating || strokePaths.length === 0) return null

    return (
      <div className="absolute inset-0 pointer-events-none">
        {/* Primary glow effects for each stroke path */}
        {strokePaths.map((strokePath, pathIndex) =>
          strokePath.points.slice(0, Math.min(strokePath.points.length, 10)).map((point, pointIndex) => (
            <React.Fragment key={`glow-effects-${strokePath.id}-${pointIndex}`}>
              {/* Main stroke glow */}
              <div
                className="stroke-glow"
                style={{
                  left: point.x - 40,
                  top: point.y - 40,
                  width: 80,
                  height: 80,
                  animationDelay: `${(pathIndex * 10 + pointIndex) * 0.05}s`
                }}
              />

              {/* Secondary pulse */}
              <div
                className="stroke-pulse"
                style={{
                  left: point.x - 30,
                  top: point.y - 30,
                  width: 60,
                  height: 60,
                  animationDelay: `${(pathIndex * 10 + pointIndex) * 0.05 + 0.3}s`
                }}
              />

              {/* Ripple wave */}
              <div
                className="ripple-wave"
                style={{
                  left: point.x - 20,
                  top: point.y - 20,
                  width: 40,
                  height: 40,
                  animationDelay: `${(pathIndex * 10 + pointIndex) * 0.05 + 0.6}s`
                }}
              />
            </React.Fragment>
          ))
        )}

        {/* Central connecting waves */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div
            className="stroke-glow"
            style={{
              width: 200,
              height: 200,
              animationDelay: '0.5s'
            }}
          />
          <div
            className="stroke-pulse"
            style={{
              width: 150,
              height: 150,
              animationDelay: '1s'
            }}
          />
        </div>
      </div>
    )
  }

  return (
    <div className={`fixed inset-0 ${isDarkMode ? 'canvas-dark' : 'canvas-light'} cursor-crosshair`}>
      <canvas
        ref={canvasRef}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        className="absolute inset-0"
        style={{ touchAction: 'none' }}
      />

      {renderGlowEffects()}

      {/* Animation message */}
      {isAnimating && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-20">
          <div className="search-overlay">
            {/* Pulsing rings */}
            <div className="search-pulse"></div>
            <div className="search-pulse"></div>
            <div className="search-pulse"></div>

            {/* Message */}
            <div className="search-message">
              {currentMessage}
            </div>
          </div>
        </div>
      )}

    </div>
  )
}

export default ImmersiveCanvas
