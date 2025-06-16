import React from 'react';

import "../assets/css/HomePage.css";

const HomePage = () => {
  return (
    <div className="home-container">

      <nav className="navbar">

        <div className="nav-item">
          <span className="nav-title">Facial</span>
          <span className="nav-title">Expression</span>
          <span className="nav-title">Recognition</span>
        </div>

        <div className="nav-about">About Us</div>

      </nav>
      
      <main className="main-content"><br /><br />
        
        <div className="title-container">
          <h1 className="main-title">FEXTORFLOW</h1>
        </div>
        
        <div className="definition-section">

          <div className="definition-dot"></div>

          <div className="definition">
            <h3 className="definition-title">Expression (n.)</h3>
            <p className="definition-text">
              the act of saying what you think<br />
              or showing how you feel using<br />
              words or actions
            </p>
          </div>

        </div>
        
        <button className="start-button">Let's Begin</button>
      </main>
    </div>
  );
};

export default HomePage;