import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './Home';
import General from './general/General_Home';
import Core_gen from './general/core_general';
import Advanced from './advanced/Advanced_Home';
import Model1 from './advanced/Model_1';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/general" element={<General />} />
        <Route path="/advanced" element={<Advanced />} />
        <Route path="/general/core_general" element={<Core_gen />} />
        <Route path="/advanced/Model_1" element={<Model1 />} />
      </Routes>
    </Router>
  );
}

export default App;