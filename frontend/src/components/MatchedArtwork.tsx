import React, { useRef, useEffect, useState } from 'react'

interface MatchedArtworkProps {
  imageUrl: string
  contourPoints: number[][]
  isDarkMode: boolean
  onError?: (error: string) => void
}

const MatchedArtwork: React.FC<MatchedArtworkProps> = ({
  imageUrl,
  contourPoints,
  isDarkMode,
  onError
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const [imageLoaded, setImageLoaded] = useState(false)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })

  // Calculate random position and size on mount
  useEffect(() => {
    const calculatePositionAndSize = () => {
      const viewportWidth = window.innerWidth
      const viewportHeight = window.innerHeight

      // Reserve 10% padding on all sides
      const maxWidth = viewportWidth * 0.8
      const maxHeight = viewportHeight * 0.8

      // Random size between 40% and 70% of max dimensions
      const sizeMultiplier = 0.4 + Math.random() * 0.3
      const targetMaxWidth = maxWidth * sizeMultiplier
      const targetMaxHeight = maxHeight * sizeMultiplier

      setDimensions({ width: targetMaxWidth, height: targetMaxHeight })

      // Random position within bounds (centered bias with slight randomness)
      const xVariance = viewportWidth * 0.15 * (Math.random() - 0.5)
      const yVariance = viewportHeight * 0.15 * (Math.random() - 0.5)

      setPosition({
        x: viewportWidth / 2 + xVariance,
        y: viewportHeight / 2 + yVariance
      })
    }

    calculatePositionAndSize()

    // Recalculate on window resize
    window.addEventListener('resize', calculatePositionAndSize)
    return () => window.removeEventListener('resize', calculatePositionAndSize)
  }, [])

  // Draw contour overlay when image loads
  useEffect(() => {
    if (!imageLoaded || !imageRef.current || !overlayCanvasRef.current) {
      return
    }

    const image = imageRef.current
    const canvas = overlayCanvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Get actual rendered dimensions
    const renderedWidth = image.offsetWidth
    const renderedHeight = image.offsetHeight
    const naturalWidth = image.naturalWidth
    const naturalHeight = image.naturalHeight

    if (naturalWidth === 0 || naturalHeight === 0) {
      return
    }

    // Set canvas to match rendered image size
    canvas.width = renderedWidth
    canvas.height = renderedHeight

    // Calculate scale factors
    const scaleX = renderedWidth / naturalWidth
    const scaleY = renderedHeight / naturalHeight

    console.log('Image dimensions:', {
      rendered: { width: renderedWidth, height: renderedHeight },
      natural: { width: naturalWidth, height: naturalHeight },
      scale: { x: scaleX, y: scaleY }
    })

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw scaled contour
    if (contourPoints && contourPoints.length > 0) {
      ctx.strokeStyle = isDarkMode ? '#00ff88' : '#ff0066'
      ctx.lineWidth = 3
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.shadowColor = ctx.strokeStyle
      ctx.shadowBlur = 10

      ctx.beginPath()

      // Scale first point
      const firstPoint = contourPoints[0]
      ctx.moveTo(firstPoint[0] * scaleX, firstPoint[1] * scaleY)

      // Draw path with scaled coordinates
      for (let i = 1; i < contourPoints.length; i++) {
        const point = contourPoints[i]
        ctx.lineTo(point[0] * scaleX, point[1] * scaleY)
      }

      // Close the contour
      ctx.closePath()
      ctx.stroke()

      console.log('Drew contour with', contourPoints.length, 'points (scaled)')
    }
  }, [imageLoaded, contourPoints, isDarkMode])

  const handleImageLoad = () => {
    console.log('Image loaded successfully:', imageUrl)
    setImageLoaded(true)
  }

  const handleImageError = (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
    console.error('Image failed to load:', imageUrl)
    console.error('Image error event:', e)
    onError?.(`Failed to load image from: ${imageUrl}`)
  }

  return (
    <div
      ref={containerRef}
      className="absolute pointer-events-none z-20"
      style={{
        left: position.x,
        top: position.y,
        transform: 'translate(-50%, -50%)',
        maxWidth: `${dimensions.width}px`,
        maxHeight: `${dimensions.height}px`,
      }}
    >
      <div className="relative">
        <img
          ref={imageRef}
          src={imageUrl}
          alt="Matched artwork"
          className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
          style={{
            maxWidth: `${dimensions.width}px`,
            maxHeight: `${dimensions.height}px`,
          }}
          onLoad={handleImageLoad}
          onError={handleImageError}
          crossOrigin="anonymous"
        />

        {/* Canvas overlay for contour highlighting */}
        {imageLoaded && contourPoints && contourPoints.length > 0 && (
          <canvas
            ref={overlayCanvasRef}
            className="absolute top-0 left-0 pointer-events-none rounded-lg"
            style={{
              width: '100%',
              height: '100%'
            }}
          />
        )}
      </div>
    </div>
  )
}

export default MatchedArtwork
