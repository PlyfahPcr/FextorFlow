import { useEffect, useRef, useState } from 'react';
import './core_general.css'
import { useNavigate } from 'react-router-dom';

function Core_gen() {

  const navigate = useNavigate();

  return (
    <div className='container-1'>
        <div className='use-ftf2'>
            <div className='wrapper'>
                <div className='box1'>
                  <h3>Fextorflow</h3>
                  <div className='button-5' onClick={() => navigate(-1)}>
                    <a href="#" className="previous">&#8249;</a>
                  </div>
                </div>
                <div className='box2'>
                  <div className='button-2'>
                    choose you're image
                  </div>
                  <div className='button-3'>
                    choose you're video
                  </div>
                  <div className='button-4'>
                    webcam
                  </div>
                </div>
                <div className='box3'>
                </div>
                <div className='box4'>
                  <p>{'Happy:\nFearful:\nSadness:\nDisgusted:\nNeutral:\nAngry:\nSurprised:'}</p>
                </div>
            </div>
        </div>
    </div>
  )
}

export default Core_gen;