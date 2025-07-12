import React, { useState, useRef, useEffect, useCallback } from 'react';
import Popup from './components/Popup'

const Home = () => {
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isMouseInGrid, setIsMouseInGrid] = useState(false);
  const [trailCells, setTrailCells] = useState([]);
  const [gridDimensions, setGridDimensions] = useState({ cols: 0, rows: 0 });
  const gridRef = useRef(null);

  const CELL_SIZE = 60;

  useEffect(() => {
    const updateGridDimensions = () => {
      const cols = Math.floor(window.innerWidth / CELL_SIZE);
      const rows = Math.floor(window.innerHeight / CELL_SIZE);
      setGridDimensions({ cols, rows });
    };

    updateGridDimensions();
    window.addEventListener('resize', updateGridDimensions);
    
    return () => {
      window.removeEventListener('resize', updateGridDimensions);
    };
  }, []);

  // Get current cell position
  const getCurrentCell = useCallback((x, y) => {
    const col = Math.floor(x / CELL_SIZE);
    const row = Math.floor(y / CELL_SIZE);
    return { col, row, id: `${row}-${col}` };
  }, []);

  // Handle mouse events
  const handleMouseMove = useCallback((e) => {
    if (gridRef.current) {
      const rect = gridRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      setMousePos({ x, y });

      const currentCell = getCurrentCell(x, y);
      
      const maxRadius = 1;

      for (let dx = -maxRadius; dx <= maxRadius; dx++) {
        for (let dy = -maxRadius; dy <= maxRadius; dy++) {
          const col = currentCell.col + dx;
          const row = currentCell.row + dy;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const falloff = Math.max(0, 1 - dist / (maxRadius + 1));

          if (col >= 0 && row >= 0 && col < gridDimensions.cols && row < gridDimensions.rows) {
            const id = `${row}-${col}`;
            setTrailCells(prev => {
              const exists = prev.find(cell => cell.id === id);
              if (!exists) {
                return [...prev, {
                  id,
                  col,
                  row,
                  timestamp: Date.now(),
                  intensity: falloff
                }];
              }
              return prev;
            });
          }
        }
      }
    }
  }, [getCurrentCell, gridDimensions]);

  const handleMouseEnter = useCallback(() => {
    setIsMouseInGrid(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsMouseInGrid(false);
  }, []);

  // Fade out trail cells
  useEffect(() => {
    const interval = setInterval(() => {
      setTrailCells(prev => {
        return prev
          .map(cell => ({
            ...cell,
            intensity: Math.max(0, cell.intensity - 0.02)
          }))
          .filter(cell => cell.intensity > 0.05);
      });
    }, 80);

    return () => clearInterval(interval);
  }, []);

    const [buttonPopup, setButtonPopup] = useState(false)

  return (
    <div style={{
      height: '100vh',
      width: '100vw',
      position: 'relative',
      backgroundColor: 'black',
      overflow: 'hidden'
    }}>
      {/* Grid Background Pattern */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundSize: '60px 60px',
        backgroundImage: `
          linear-gradient(to right, rgba(124, 255, 10, 0.2) 1px, transparent 1px),
          linear-gradient(to bottom, rgba(124, 255, 10, 0.2) 1px, transparent 1px)
        `,
        opacity: 0.3
      }} />
      
      {/* Subtle radial fade */}
      <div style={{
        pointerEvents: 'none',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'radial-gradient(ellipse at center, transparent 30%, rgba(0, 0, 0, 0.8) 100%)'
      }} />

      {/* Interactive Grid Layer */}
      <div 
        ref={gridRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          cursor: 'default',
          userSelect: 'none',
          zIndex: 10
        }}
        onMouseMove={handleMouseMove}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {/* Trail cells */}
        {trailCells.map((cell) => (
          <div
            key={`${cell.id}-${cell.timestamp}`}
            style={{
              position: 'absolute',
              pointerEvents: 'none',
              left: cell.col * CELL_SIZE + 1,
              top: cell.row * CELL_SIZE + 1,
              width: CELL_SIZE - 2,
              height: CELL_SIZE - 2,
              backgroundColor: 'transparent',
              border: `1px solid rgba(124, 255, 10, ${cell.intensity * 0.8})`,
              boxShadow: `
                0 0 ${cell.intensity * 25}px rgba(124, 255, 10, ${cell.intensity * 0.6}),
                inset 0 0 ${cell.intensity * 12}px rgba(124, 255, 10, ${cell.intensity * 0.2})
              `,
              borderRadius: '0px',
              transition: 'opacity 0.1s ease-out',
              zIndex: 15
            }}
          />
        ))}

        {/* Current cell highlight */}
        {isMouseInGrid && gridDimensions.cols > 0 && (
          <div
            style={{
              position: 'absolute',
              pointerEvents: 'none',
              left: Math.floor(mousePos.x / CELL_SIZE) * CELL_SIZE + 1,
              top: Math.floor(mousePos.y / CELL_SIZE) * CELL_SIZE + 1,
              width: CELL_SIZE - 2,
              height: CELL_SIZE - 2,
              backgroundColor: 'transparent',
              border: '2px solid rgba(124, 255, 10, 0.5)',
              boxShadow: `
                0 0 35px rgba(124, 255, 10, 0.8),
                inset 0 0 18px rgba(124, 255, 10, 0.3)
              `,
              borderRadius: '0px',
              zIndex: 20
            }}
          />
        )}

        {/* Mouse glow effect */}
        {isMouseInGrid && (
          <div
            style={{
              position: 'absolute',
              pointerEvents: 'none',
              left: mousePos.x - 50,
              top: mousePos.y - 50,
              width: 100,
              height: 100,
              background: 'radial-gradient(circle, rgba(124, 255, 10, 0.25) 0%, rgba(124, 255, 10, 0.08) 30%, transparent 70%)',
              borderRadius: '50%',
              zIndex: 5
            }}
          />
        )}
      </div>
   
      {/*Logo*/ }
      <div className="logo">         
        <div className="fextorflow">
          Fextorflow
        </div>                
      </div> 

      <div className="center">
      <p>{'FACIAL\nEXPRESSION\nRECOGNITION'}</p>
      </div>
      
      <div className="pop">
          <button className="button" onClick={() => setButtonPopup(true)}>Let's Begin</button>
          <Popup trigger={buttonPopup} setTrigger={setButtonPopup}>
          </Popup>
      </div>

    </div>
    
  );
};

export default Home;