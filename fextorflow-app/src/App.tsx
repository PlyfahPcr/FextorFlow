import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './Home';
import General from './general/General_Home';
import Core_gen from './general/core_general';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/general" element={<General />} />
        <Route path="/general/core_general" element={<Core_gen />} />
      </Routes>
    </Router>
  );
}

export default App;