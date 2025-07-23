import { useEffect, useRef, useState } from 'react';
import './general/home.css'
import Popup from './components/Popup'
import { useNavigate } from 'react-router-dom';

function Home(props) {
  const navigate = useNavigate();

  const handleCoreGeneralUser = () => {
    navigate('/general/core_general');
    props.setTrigger(false);
  };
  
  const handleHome = () => {
    navigate('/');
    props.setTrigger(false);
  };

  const [buttonPopup, setButtonPopup] = useState(false)

  return (
    <div className="container-1">
        <div className='page-h'>
            <div className='custom-grid-h'> 
                <div className="s1-h"></div>
                <div className="logo-h">         
                    <div className="fextorflow-h" onClick={handleHome}>
                    Fextorflow
                    </div>           
                </div> 
                <div className="s2-h"></div>
                <div className="s3-h"></div>
                <div className="s4-h"></div>
                <div className="s5-h"></div>
                <div className='h-text'>
                    <p>{'Expression (n.)\nthe act of saying what you think or\n showing how you feel using words or actions'}</p>
                </div>
            </div>
        </div>
        <div className="container-2">
            <div className='use-h'>
              <h2>{'FACIAL\nEXPRESSION\nRECOGNITION'}</h2>
              <p>{'is a computer vision task aimed at identifying and\ncategorizing emotional expressions depicted on a human face.\n The goal is to automate the process of determining emotions \nin real-time, by analyzing the various features of a face\n such as eyebrows, eyes, mouth, and other features, and\n mapping them to a set of emotions such as anger, fear, \nsurprise, sadness, happiness and neutral.'}</p>
            </div>
            
            <div className="role">
              <h3>Roles for you</h3>
            </div>
            
            <div className='role_types'>
              <div className='role_box1'>
                <h3>GENERAL USER</h3>
                <p>{'Users who want to explore or are\n interested in learning about facial\n expression recognition'}</p>
              </div>
              <div className='role_box2'>
                <h3>ADVANCED USER</h3>
                <p>{'Users who need comprehensive access to\n the system\'s capabilities for their\n specialized work in facial expression\n recognition'}</p>
              </div>
            </div>
            <button className="button-h" onClick={() => setButtonPopup(true)}>Let's Begin</button>
            <Popup trigger={buttonPopup} setTrigger={setButtonPopup}>
            </Popup>
            <img src="/public/img/left.png" alt="left arrow" className="L-arrow" />
            <img src="/public/img/right.png" alt="right arrow" className="R-arrow" />
        </div>
        
        </div>
  );
}

export default Home;