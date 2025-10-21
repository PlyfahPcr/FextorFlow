import { useEffect, useRef, useState } from 'react';
import './Model.css'
import { useNavigate } from 'react-router-dom';
import Core_Ad from './core_advanced';

function Tech_Info(props) {

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
    navigate('/advanced/Model_1');
    props.setTrigger(false);
  };

  return (
    <div className="container-m">
        <div className='page-m'>  
          <div className="home-nav">
            <span className="home-nav-item" onClick={handleHome}>HOME</span>
            <span className="home-nav-item" onClick={handleBack}>BACK</span>
            <span className="home-nav-item" onClick={handleExplore}>COMPARATIVE</span>
          </div>
            <div className="logo-m">         
                <div className="fextorflow-m" onClick={handleHome}>
                Fextorflow
                </div>           
            </div> 
        </div> 
        <div className='m-name'>
          <p> Model 1 </p>
        </div>  
        <img src="/public/img/model-graph.png" className="m-graph" /> 
    </div>
   
  );
}

export default Tech_Info;