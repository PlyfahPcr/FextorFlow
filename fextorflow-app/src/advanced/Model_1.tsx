import { useEffect, useRef, useState } from 'react';
import './Model.css'
import { useNavigate } from 'react-router-dom';
import Core_Ad from './core_advanced';

function model1(props) {
{/*
const section2Ref = useRef(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsVisible(entry.isIntersecting);
      },
      { threshold: 0.6 }
    );

    if (section2Ref.current) {
      observer.observe(section2Ref.current);
    }

    return () => {
      if (section2Ref.current) {
        observer.unobserve(section2Ref.current);
      }
    };
  }, []);  */}

  const navigate = useNavigate();

  const handleHome = () => {
        navigate('/');
        props.setTrigger(false);
    };

  const handleExplore = () => {
    navigate('/advanced/Compare');
    props.setTrigger(false);
  };

  const handleBack = () => {
    navigate('/advanced');
    props.setTrigger(false);
  };

  const handleTechInfo = () => {
    navigate('/advanced/TechInfo');
    props.setTrigger(false);
  };

  return (
    <div className="container-m">
        <div className='page-m'>  
          <div className="home-nav">
            <span className="home-nav-item" onClick={handleHome}>HOME</span>
            <span className="home-nav-item" onClick={handleBack}>MODEL</span>
            <span className="home-nav-item" onClick={handleExplore}>COMPARATIVE</span>
          </div>
            <div className="logo-m">         
                <div className="fextorflow-m" onClick={handleHome}>
                Fextorflow
                </div>           
            </div> 
            <Core_Ad></Core_Ad>
        </div> 
        
        {/*
        <div className='m-name'>
          <p> Model 1 </p>
        </div>  
        <img src="/public/img/model-graph.png" className="m-graph" /> 
        <div className={`m-compare ${isVisible ? 'hidden' : ''}`}>*/}
        <img src="/img/circle.png" className='circle-img'/>
        <img src="/img/ar-down.png" className='ardown-img'/>
        <div className='m-compare'>
          <p>{'Technical\n Information '}</p>
        </div>
        <div className={`button-compare` } onClick={handleTechInfo}>
          <p>Explore more</p>
        </div>
        {/*
        <div className={`overlay-m ${isVisible ? 'show' : ''}`} ref={section2Ref}>
        </div>*/}
    </div>
   
  );
}

export default model1;