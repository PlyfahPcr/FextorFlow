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
  
  const words = ["HAPPY", "SAD", "NEUTRAL", "DISGUSTING", "SURPRISE", "ANGRY", "FEAR"];
  const color = {
    HAPPY: "#ffc934ff",
    SAD: "#519fccff",
    NEUTRAL: "#cfd1d6",
    DISGUSTING: "#538340",
    SURPRISE: "#ffa947",
    ANGRY: "#db2929",
    FEAR: "#9158d1ff"
  };

  const [index, setIndex] = useState(0);
  const [fade, setFade] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setFade(true);
      setTimeout(() => {
        setIndex((prev) => (prev + 1) % words.length);
        setFade(false);
      }, 600);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const currentWord = words[index];
  const currentColor = color[currentWord];

  return (
    <div className="container-1">

          <div className="nav">         
            <div className="nav-item" onClick={handleHome}>
              Fextorflow
            </div>        
          </div> 
          <div  className={`emotions ${fade ? 'fade' : '1'}`}
          style={{ color: currentColor }}>
            {currentWord}
          </div>
          <button className="button-h" onClick={() => setButtonPopup(true)}>Find your ROLE</button>
          <Popup trigger={buttonPopup} setTrigger={setButtonPopup}></Popup>  
          <div className='h-text'>
            <p>{'Expression (n.)\nthe act of saying what you think or\n showing how you feel using words or actions'}</p>
          </div>
      </div>
  );
}

export default Home;