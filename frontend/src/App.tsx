import React, { useState, useCallback } from 'react'
import ImmersiveCanvas from './components/ImmersiveCanvas'
import FloatingControls from './components/FloatingControls'
import './App.css'

function App() {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('pastiche-dark-mode')
    return saved ? JSON.parse(saved) : false // Default to light mode
  })

  const [strokeCount, setStrokeCount] = useState(0)
  const [isSearching, setIsSearching] = useState(false)
  const maxStrokes = 25

  // Toggle dark mode
  const toggleDarkMode = useCallback(() => {
    setIsDarkMode((prev: boolean) => {
      const newValue = !prev
      localStorage.setItem('pastiche-dark-mode', JSON.stringify(newValue))
      return newValue
    })
  }, [])

  // Clear canvas
  const handleClear = useCallback(() => {
    setStrokeCount(0)
    setIsSearching(false)
  }, [])

  // Handle stroke count changes from canvas
  const handleStrokeCountChange = useCallback((count: number) => {
    setStrokeCount(count)
  }, [])

  // Handle search trigger
  const handleSearchTrigger = useCallback(() => {
    if (strokeCount > 0) {
      setIsSearching(true)
      // Auto-hide search after animation completes
      setTimeout(() => setIsSearching(false), 4000)
    }
  }, [strokeCount])

  return (
    <div className={`app ${isDarkMode ? 'dark' : 'light'}`}>
      {/* Immersive Canvas */}
      <ImmersiveCanvas
        isDarkMode={isDarkMode}
        onStrokeCountChange={handleStrokeCountChange}
        onSearchTrigger={handleSearchTrigger}
        onClear={handleClear}
        maxStrokes={maxStrokes}
        isSearching={isSearching}
      />

      {/* Floating Controls */}
      <FloatingControls
        isDarkMode={isDarkMode}
        onToggleDarkMode={toggleDarkMode}
      />
    </div>
  )
}

export default App
