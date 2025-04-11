// src/pages/LandingPage.tsx
import React from 'react';

import Slider from 'react-slick';
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';
import '../assets/styles/carousel.css';
import { useNavigate } from 'react-router-dom';

// Styles impprts
import '../index.css';
import '../assets/styles/landingpage.css';

// Component imports
import NavBar from '../components/NavBar';
import Footer from '../components/Footer';
import Button from '../components/Button';


// Using vite to load all images in the folder.
const imageModules = import.meta.glob('../assets/images/carousel/*.webp', {
  eager: true,
  import: 'default',
});

// Converting the object into an array
const carouselImages = Object.keys(imageModules).map((path) => {
  const src = imageModules[path] as string;
  const alt = path.split('/').pop()?.split('.')[0] ?? 'Image';
  return {src, alt}; 
  // Extracts the image path and filename
});

const LandingPage: React.FC = () => {
  // Page specific logic goes here:

  // Settings to define the behaviour of the carousel
  const settings = {
    infinite : true,
    slidesToShow: 1,
    autoplay: true, 
    autoplaySpeed: 2000,
    arrows: false,
  };

  const navigate = useNavigate();

  const handleClick = () => {
    navigate('/valuation');
  }


  return (

    <body>
      <NavBar/>
      <div className = "home-page">

        <header>
          <h1 className = "cyber-heading animate-fade-in-down-1" >Value Your car</h1>

        </header>
        
        <hr></hr>

        <div className = "slider-wrapper">
          <div className = "desc-text animate-fade-in-down-2">
              Decrypt your vehicle's true market value in seconds.
              Our quantum algorithm analyses real-time market data, 
              vehicle specs, and regional variables to bypass traditional valuation barriers.
          </div>
        {/* Passing the settings object to the slider */}
        <Slider {...settings}>
          {carouselImages.map((image, index) => (
                <div className= 'carousel-item' key = {index}>
                  <img src={image.src} alt={image.alt} />
                </div>
              ))}
        </Slider>
        </div> 
    
          {/*Tailwind to make the button centered, the style is in the button component*/}
        <div className='flex justify-center mt-10 pb-40'>
          <Button label = "Value Your Car Now" onClick={handleClick}/>
        </div>
      </div>
      <Footer/>
    </body>

  );
};

export default LandingPage;
