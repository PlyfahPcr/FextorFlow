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

  const [file, setFile] = useState(null);
  const fileInputRef = useRef(null);
  const videoInputRef = useRef(null);
  
    function handleChange(e) {
        console.log(e.target.files);
        setFile(URL.createObjectURL(e.target.files[0]));
    }
  
    function handleVideoChange(e) {
        console.log(e.target.files);
        setFile(URL.createObjectURL(e.target.files[0]));
    }
  
    const handleImportClick = () => {
        fileInputRef.current.click();
    };
  
    const handleVideoClick = () => {
        videoInputRef.current.click();
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
                  <div className='button-2ad' onClick={handleImportClick}>
                    Import Image
                    <img src="/img/add-image.png" className='ad_add_img'/>
                    <input 
                        type="file" 
                        ref={fileInputRef}
                        onChange={handleChange} 
                        style={{ display: 'none' }}
                      />
                  </div>
                  <div className='button-3ad' onClick={handleVideoClick}>
                    Import video
                    <img src="/img/video-production.png" className='ad_add_img'/>
                    <input 
                        type="file" 
                        ref={videoInputRef}
                        onChange={handleVideoChange} 
                        style={{ display: 'none' }}
                      />
                  </div>
                  <div className='button-4ad'>
                    Webcam
                    <img src="/img/webcam.png" className='ad_add_img_wc'/>
                  </div>
                </div>
                <div className='box3-ad'>
                  {file && (
                    file.includes('video') || file.includes('.mp4') || file.includes('.mov') || file.includes('.avi') ? 
                    <video src={file} controls style={{ maxWidth: '100%', height: 'auto' }} /> :
                    <img src={file} alt="Uploaded preview" style={{ maxWidth: '100%', height: 'auto' }} />
                  )}
                </div>
                <div className='box4-ad'>
                  <div className="person-ad">
                    <div className="person-ad__head">
                      <div className='person-ad__head2'></div>
                      <div className='person-ad__face'>
                      </div>
                    </div>
                  </div>
                </div>
            </div>
        </div>
    </div>
  )
}

export default Core_Ad;