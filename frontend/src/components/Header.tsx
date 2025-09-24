import React from 'react'
import './Header.css'

const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <h1>Pastiche</h1>
          <span className="tagline">Sketch to Artwork Discovery</span>
        </div>
        <nav className="nav">
          <button className="nav-btn">Gallery</button>
          <button className="nav-btn">About</button>
          <button className="nav-btn primary">Find Matches</button>
        </nav>
      </div>
    </header>
  )
}

export default Header
