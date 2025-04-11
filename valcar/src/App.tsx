// src/App.tsx
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import ValuationForm from './pages/ValuationForm';

const App: React.FC = () => {
  return (
    <Routes>
      {/* adding more routes: <Route path="/contact" element={<ContactPage />} /> */}
      <Route path="/" element={<LandingPage />} />
      <Route path="/valuation" element={<ValuationForm/>} />
    </Routes>
  );
};

export default App;
