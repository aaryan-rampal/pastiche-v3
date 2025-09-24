import React from 'react'

interface FloatingControlsProps {
  isDarkMode: boolean
  onToggleDarkMode: () => void
  onClear: () => void
  onUpload: () => void
  strokeCount: number
  maxStrokes: number
}

const FloatingControls: React.FC<FloatingControlsProps> = ({
  isDarkMode,
  onToggleDarkMode,
}) => {
  return (
    <>
      {/* Top Controls */}
      <div className="absolute top-6 right-6 flex items-center gap-4">
        {/* Dark Mode Toggle */}
        <button
          onClick={onToggleDarkMode}
          className="glass-control bg-white/10 hover:bg-white/20 transition-all duration-300 rounded-full p-3"
          aria-label={`Switch to ${isDarkMode ? 'light' : 'dark'} mode`}
        >
          {isDarkMode ? (
            <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
            </svg>
          ) : (
            <svg className="w-5 h-5 text-gray-800" fill="currentColor" viewBox="0 0 20 20">
              <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
            </svg>
          )}
        </button>
      </div>

      {/* Pastiche info section */}
      {/* Make it smaller */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
        <span className={`font-medium text-gray-400`}>
        <center><p><strong>Pastiche</strong></p></center>
        <p><small><em>"An artistic work in a style that imitates that of another work, artist, or period."</em></small></p>
        </span>
      </div>

      {/* Help Text */}
      <div className="absolute bottom-4 right-4">
        <div className="glass-control bg-white/5 rounded-lg px-3 py-2 text-xs">
          <p className={`${isDarkMode ? 'text-white/60' : 'text-gray-600'}`}>
            Go on, draw something...
          </p>
        </div>
      </div>
    </>
  )
}

export default FloatingControls
