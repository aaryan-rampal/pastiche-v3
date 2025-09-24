import React, { useRef, useEffect, useState, useCallback } from 'react'

interface Stroke {
  x: number
  y: number
  pressure: number
  angle: number
  id: number
}

interface ImmersiveCanvasProps {
  isDarkMode: boolean
}

const ImmersiveCanvas: React.FC<ImmersiveCanvasProps> = ({ isDarkMode }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const contextRef = useRef<CanvasRenderingContext2D | null>(null)
  const [strokes, setStrokes] = useState<Stroke[]>([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [isAnimating, setIsAnimating] = useState(false)
  const [currentMessage, setCurrentMessage] = useState('')

  const maxStrokes = 25
  const strokeIdRef = useRef(0)
  const lastMousePos = useRef({ x: 0, y: 0 })
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

  // Render strokes with brush-like effect
  useEffect(() => {
    const canvas = canvasRef.current
    const context = contextRef.current
    if (!canvas || !context) return

    // Clear canvas
    context.clearRect(0, 0, canvas.width, canvas.height)

    // Draw each stroke with brush effect
    strokes.forEach((stroke) => {
      context.save()

      // Set stroke color based on theme
      const strokeColor = isDarkMode ? '#f5f2ed' : '#1a1614'

      // Create brush-like effect
      const brushSize = 4 + stroke.pressure * 6

      // Main stroke
      context.fillStyle = strokeColor
      context.beginPath()
      context.arc(stroke.x, stroke.y, brushSize, 0, Math.PI * 2)
      context.fill()

      // Add texture with multiple smaller circles
      for (let i = 0; i < 5; i++) {
        const offsetX = (Math.random() - 0.5) * brushSize * 1.5
        const offsetY = (Math.random() - 0.5) * brushSize * 1.5
        const size = brushSize * (0.3 + Math.random() * 0.4)

        context.globalAlpha = 0.6 + Math.random() * 0.4
        context.beginPath()
        context.arc(stroke.x + offsetX, stroke.y + offsetY, size, 0, Math.PI * 2)
        context.fill()
      }

      // Add directional streaks
      context.globalAlpha = 0.4
      context.fillStyle = strokeColor
      const streakLength = brushSize * 2
      const streakWidth = brushSize * 0.3

      context.save()
      context.translate(stroke.x, stroke.y)
      context.rotate(stroke.angle)
      context.fillRect(-streakLength/2, -streakWidth/2, streakLength, streakWidth)
      context.restore()

      context.restore()
    })
  }, [strokes, isDarkMode])

  // Start drawing
  const startDrawing = useCallback((event: React.MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    lastMousePos.current = { x, y }
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

    // Calculate angle from mouse movement
    const dx = x - lastMousePos.current.x
    const dy = y - lastMousePos.current.y
    const angle = Math.atan2(dy, dx)

    // Calculate pressure from movement speed (simulated)
    const distance = Math.sqrt(dx * dx + dy * dy)
    const pressure = Math.min(1, Math.max(0.3, 1 - distance / 50))

    const newStroke: Stroke = {
      x,
      y,
      pressure,
      angle,
      id: strokeIdRef.current++
    }

    setStrokes(prev => {
      const updated = [...prev, newStroke]
      // Keep only the latest 25 strokes (instant removal)
      return updated.slice(-maxStrokes)
    })

    lastMousePos.current = { x, y }
  }, [isDrawing])

  // Stop drawing and trigger animation
  const stopDrawing = useCallback(() => {
    if (!isDrawing) return

    setIsDrawing(false)

    const messages = [
      "Finding the perfect artwork...",
      "Discovering masterpiece matches...",
      "Connecting with art history...",
      "Searching the masters...",
      "Exploring artistic connections...",
      "Unveiling hidden similarities..."
    ]

    // Start animation after 2 seconds of inactivity
    animationTimeoutRef.current = setTimeout(() => {
      if (strokes.length > 0) {
        setIsAnimating(true)
        setCurrentMessage(messages[Math.floor(Math.random() * messages.length)])
      }
    }, 2000)
  }, [isDrawing, strokes.length])

  // Render glow effects during animation
  const renderGlowEffects = () => {
    if (!isAnimating || strokes.length === 0) return null

    return (
      <div className="absolute inset-0 pointer-events-none">
        {strokes.map((stroke, index) => (
          <div
            key={`glow-${stroke.id}`}
            className="absolute amber-glow rounded-full"
            style={{
              left: stroke.x - 30,
              top: stroke.y - 30,
              width: 60,
              height: 60,
              background: `radial-gradient(circle, rgba(245, 158, 11, 0.3) 0%, rgba(245, 158, 11, 0.1) 50%, transparent 100%)`,
              animationDelay: `${index * 0.1}s`
            }}
          />
        ))}
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
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="bg-black/20 backdrop-blur-md rounded-full px-8 py-4">
            <p className={`text-lg font-medium ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
              {currentMessage}
            </p>
          </div>
        </div>
      )}

      {/* Stroke counter */}
      <div className="absolute top-6 left-6 glass-control bg-white/10 rounded-full px-4 py-2">
        <span className={`text-sm font-medium ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
          {strokes.length}/{maxStrokes} strokes
        </span>
      </div>
    </div>
  )
}

export default ImmersiveCanvas
