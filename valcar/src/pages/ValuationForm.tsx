import React from 'react';

// Component Import
import NavBar from '../components/NavBar';
import Footer from '../components/Footer';

const ValuationForm: React.FC = () => {
  return (
    <div>
      <NavBar/>
      <div className='pb-300'>
        input form here
      </div>
      <Footer/>
    </div>

  );
};

export default ValuationForm;