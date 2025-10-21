import React from 'react';
import './Popup.css'
import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';



function Popup(props) {

  const navigate = useNavigate();

  const handleGeneralUser = () => {
    navigate('/general');
    props.setTrigger(false);
  };

  const handleAdvancedUser = () => {
    navigate('/advanced');
    props.setTrigger(false);
  };

  const handleInformation = () => {
    navigate('/general/Home_information');
    props.setTrigger(false);
  };

  return (props.trigger) ? (
    <div className="popup">
        <div className="popup-inner">
          <div className="info-wrapper">
            <img src="/img/info.png" className="info"  onClick={handleInformation }/>
            <div className="hide">More Information</div>
          </div>
            <div className="outer">
                <button className="inner"onClick={ () => props.setTrigger(false)}>
                    <label>Close</label>
                </button>
            { props.children }
            </div>
            <div className='s3-h'>
              <h2>General User</h2>
              <p>{'Users who want to explore or are interested in learning about facial expression recognition'}</p>
              <img src="/img/angry.png" className="angry"/>
              <img src="/img/Texta.png" className="angry-text"/>
              <button className="g-button"onClick={handleGeneralUser}> Explore More</button>
            </div>
            <div className='s2-h'>
              <h2>Advanced User</h2>
              <p>{'Users who need comprehensive access to the system\'s capabilities for their specialized work in facial expression recognition'}</p>
              <img src="/img/Fear.png" className="fear"/>
              <img src="/img/Textf.png" className="fear-text"/>
              <button className="a-button"onClick={handleAdvancedUser}> Explore More</button>
            </div>

        </div>
    </div>
  ): "";
}

export default Popup;
