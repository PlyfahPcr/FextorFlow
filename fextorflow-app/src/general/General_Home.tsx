import { useEffect, useRef, useState } from 'react';
import './General_Home.css'
import { useNavigate } from 'react-router-dom';

function General(props) {
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
  }, []);

  const navigate = useNavigate();

  const handleCoreGeneralUser = () => {
    navigate('/general/core_general');
    props.setTrigger(false);
  };
  const handleHome = () => {
        navigate('/');
        props.setTrigger(false);
    };
  return (
    <div className="container">
        <div className='page'>
            <div className='custom-grid'>
              <div className="home-1" onClick={handleHome}>
                        HOME
                </div>   
            <div className="s1"></div>
            <div className="logo-g">         
                <div className="fextorflow-g" onClick={handleHome}>
                Fextorflow
                </div>
                             
            </div> 
            <div className="s2"></div>
            <div className="s3"></div>
            <div className="s4"></div>
            <div className="s5"></div>
            <div className='E-text'>
                <p>{'Expression (n.)\nthe act of saying what you think or\n showing how you feel using words or actions'}</p>
            </div>
            </div>
        </div>
        <div className={`overlay ${isVisible ? 'show' : ''}`}ref={section2Ref}>
            <div className='use-ftf'>
                <h3>How to use our Fextorflow:</h3>
                <p>{'Where does it come from?\nContrary to popular belief, Lorem Ipsum is not simply random text.\n It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old.\n Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source.\n Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC.\n This book is a treatise on the theory of ethics, very popular during the Renaissance.\n The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.\nThe standard chunk of Lorem Ipsum used since the 1500s is reproduced below for those interested.\n Sections 1.10.32 and 1.10.33 from "de Finibus Bonorum et Malorum" by Cicero are also reproduced in their exact original form, accompanied by English versions from the 1914 translation by H. Rackham.'}</p>
            </div>
            <button className="button-1" onClick={handleCoreGeneralUser}>Start</button>
        </div>
        
            
   
    </div>
   
  );
}

export default General;