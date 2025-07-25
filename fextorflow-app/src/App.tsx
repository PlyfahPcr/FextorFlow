import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './Home';
import General from './general/General_Home';
import Core_gen from './general/core_general';
import Advanced from './advanced/Advanced_Home';
import Model1 from './advanced/Model_1';
import Compare from './advanced/Compare';
import ViewResult from './advanced/View_Result';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/general" element={<General />} />
        <Route path="/advanced" element={<Advanced />} />
        <Route path="/general/core_general" element={<Core_gen />} />
        <Route path="/advanced/Model_1" element={<Model1 />} />
        <Route path="/advanced/Compare" element={<Compare />} />
        <Route path="/advanced/View_Result" element={<ViewResult />} />
      </Routes>
    </Router>
  );
}

export default App;