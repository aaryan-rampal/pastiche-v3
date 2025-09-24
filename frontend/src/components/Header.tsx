import React from 'react'

const Header: React.FC = () => {
  return (
    <header className="bg-gradient-to-r from-amber-50 to-stone-100 border-b-4 border-amber-200 shadow-lg">
      <div className="max-w-6xl mx-auto px-8 py-8">
        <div className="text-center">
          <h1 className="font-['Playfair_Display'] text-6xl font-bold text-stone-800 mb-2 tracking-wide">
            Pastiche
          </h1>
          <div className="w-32 h-0.5 bg-gradient-to-r from-transparent via-yellow-600 to-transparent mx-auto mb-3"></div>
          <p className="text-stone-700 text-lg italic font-light">
            Where your sketches meet the masters
          </p>
        </div>
      </div>
    </header>
  )
}

export default Header
