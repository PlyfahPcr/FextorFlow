import { useEffect, useRef, useState } from 'react';
import './core_advanced.css'
import { useNavigate } from 'react-router-dom';

function Core_Ad(props) {

  const navigate = useNavigate();
  const handleViewResult = () => {
    // Fixed navigation path - since View_Result.tsx is in the same directory as core_advanced.tsx
    navigate('/advanced/View_Result');
    // Only call setTrigger if it exists in props
    if (props?.setTrigger) {
      props.setTrigger(false);
    }
  };

  return (
    <div className='container-ad'>
        <div className='use-ad'>
            <div className='wrapper-ad'>
                <div className='box1-ad'>
                  <h3>Fextorflow</h3>
                  <div className='button-5ad' onClick={handleViewResult }>
                    <p>View Result</p>
                    {/*<a href="#" className="previous">&#8249;</a>*/}
                  </div>
                </div>
                <div className='box2-ad'>
                  <div className='button-2ad'>
                    Import Image
                  </div>
                  <div className='button-3ad'>
                    Import video
                  </div>
                  <div className='button-4ad'>
                    Webcam
                  </div>
                </div>
                <div className='box3-ad'>
                </div>
                <div className='box4-ad'>
                  <p>{'Face:\nPose:\nEmotion:\nPan:'}</p>
                </div>
            </div>
        </div>
    </div>
  )
}

export default Core_Ad;