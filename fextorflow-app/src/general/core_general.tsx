import { useEffect, useRef, useState } from 'react';
import './core_general.css'
import { useNavigate } from 'react-router-dom';

function Core_gen() {

  const navigate = useNavigate();

  return (
    <div className='container-3'>
        <div className='use-ftf2'>
            <div className='wrapper'>
                <div className='box1'>
                  <h3>Fextorflow</h3>
                  {/*<div className='button-5' onClick={() => navigate(-1)}>
                    <p>DOWNLAOD</p>
                    <a href="#" className="previous">&#8249;</a>
                  </div>*/}
                </div>
                <div className='box2'>
                  <div className='button-2'>
                    Import Image
                  </div>
                  <div className='button-3'>
                    Import video
                  </div>
                  <div className='button-4'>
                    Webcam
                  </div>
                </div>
                <div className='box3'>
                </div>
                <div className='box4'>
                  <p>{'Face:\nPose:\nEmotion:\nPan:'}</p>
                </div>
            </div>
        </div>
    </div>
  )
}

export default Core_gen;