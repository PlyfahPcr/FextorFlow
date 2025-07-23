import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Advanced_Home.css'

const HorizontalScrollCarousel = ({ cards }) => {
  const carouselRef = useRef(null);
  const containerRef = useRef(null);
  const originalCards = [...cards];
  const loopedCards = Array(5).fill(originalCards).flat();


  useEffect(() => {
    const handleWheel = (e) => {
      const container = containerRef.current;
      const carousel = carouselRef.current;
      
      if (!container || !carousel) return;
      
      const rect = container.getBoundingClientRect();
      const mouseX = e.clientX;
      const mouseY = e.clientY;
      
      const isInside = mouseX >= rect.left && mouseX <= rect.right && 
                      mouseY >= rect.top && mouseY <= rect.bottom;
      
      if (isInside) {
        e.preventDefault();
        
        const scrollAmount = e.deltaY * 7.0;
        carousel.scrollLeft += scrollAmount;
      }
    };

    const container = containerRef.current;
    if (container) {
      container.addEventListener('wheel', handleWheel, { passive: false });
      
      return () => {
        container.removeEventListener('wheel', handleWheel);
      };
    }
  }, []);

  return (
    <div 
      ref={containerRef}
    >
      <div
        ref={carouselRef}
        className="carousel-track"
        style={{ 
          scrollBehavior: "smooth", 
          cursor: "auto",
        }}
      >
        {loopedCards.map((card, index) => (
          <Card card={card} key={index} />
        ))}
      </div>
    </div>
  );
};

const Card = ({ card }) => {
   return (
    <div className="card" onClick={card.onClick}>
      <div
        className="card-background"
        style={{
          backgroundImage: `url(${card.url})`,
        }}
      ></div>
      <div className="card-content">
        <p className="card-title">
          {card.title}
        </p>
      </div>
    </div>
  );
};

const Advanced = (props) => {
    const navigate = useNavigate();

    const handleHome = () => {
        navigate('/');
        props.setTrigger(false);
    };

    const handleModel1 = () => {
        navigate('/advanced/Model_1');
        props.setTrigger(false);
    };

    const handleCompare = () => {
        navigate('/advanced/Compare');
        props.setTrigger(false);
    };

const cards = [
  {
    url: "#",
    title: "comparative information about\n multiple model results on\n the same dataset",
    id: 1,
    onClick: handleCompare,
  },
  {
    url: "#",
    title: "Model 1",
    id: 2,
    onClick: handleModel1,
  },
  {
    url: "#",
    title: "Model 2",
    id: 3,
  },
  {
    url: "#",
    title: "Model 3",
    id: 4,
  },
  
];
  return (
    <div className="ad">
        <div className="container-2">
            <div className="wrapper-1">
                <div className="top">
                    <div className="logo-a">         
                        <div className="fextorflow-a" onClick={handleHome}>
                        Fextorflow
                        </div>                
                    </div> 
                    <div className="home" onClick={handleHome}>
                        HOME
                    </div>
                </div>
                <div className='middle'>
                    <HorizontalScrollCarousel cards={cards} />
                </div>
                <div className="bottom">
                    <div className='text'>
                        <p>{'Expression (n.)\nthe act of saying what you think or\n showing how you feel using words or actions'}</p>
                    </div>
                    <div className='fer'>
                        <p>{'FER.'}</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
        
  );
};




export default Advanced;

