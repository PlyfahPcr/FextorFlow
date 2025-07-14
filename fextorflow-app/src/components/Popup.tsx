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

  return (props.trigger) ? (
    <div className="popup">
        <div className="popup-inner">
            <div className="outer">
                <button className="inner"onClick={ () => props.setTrigger(false)}>
                    <label>Close</label>
                </button>
            { props.children }
            </div>
            <button className="g-button"onClick={handleGeneralUser}> General User</button>
            <button className="a-button"onClick={handleAdvancedUser}> Advanced User</button>
        </div>
    </div>
  ): "";
}

export default Popup;
