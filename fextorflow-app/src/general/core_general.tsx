import React, { useEffect, useRef, useState, Suspense } from 'react';
import './core_general.css';
import { useNavigate } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import Scene from '../assets/Scene';
import Webcam from "react-webcam";
import Face from '../assets/Face';
import * as THREE from 'three';

function Core_gen() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);
  const videoInputRef = useRef(null);
  const webcamRef = useRef(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [previewList, setPreviewList] = useState<string[]>([]);
  const [panDeg, setPanDeg] = useState(0);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const captureIntervalRef = useRef(null);

  function handleChange(e) {
    console.log(e.target.files);
    if (e.target.files[0]) {
      setFile(URL.createObjectURL(e.target.files[0]));
      setIsWebcamActive(false);
      uploadImage(e.target.files[0]);
    }
  }

  function handleVideoChange(e) {
    console.log(e.target.files);
    if (e.target.files[0]) {
      setFile(URL.createObjectURL(e.target.files[0]));
      setIsWebcamActive(false);
    }
  }

  const handleImportClick = () => {
    fileInputRef.current.click();
  };

  const handleVideoClick = () => {
    videoInputRef.current.click();
  };

  const handleWebcamClick = () => {
    setIsWebcamActive(!isWebcamActive);
    setFile(null);
    setPrediction(null);
    setPreview(null);
    setPreviewList([]);
  };

  const captureWebcam = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        const response = await fetch(imageSrc);
        const blob = await response.blob();
        const file = new File([blob], "webcam-capture.jpg", { type: "image/jpeg" });
        
        setFile(imageSrc);
        await uploadImage(file);
      }
    }
  };

  // Real-time continuous capture
  useEffect(() => {
    if (isWebcamActive && !loading) {
      // เริ่ม capture ทันที
      captureWebcam();
      
      // ตั้งค่า interval เพื่อ capture ต่อเนื่อง
      captureIntervalRef.current = setInterval(() => {
        captureWebcam();
      }, 500); // capture ทุก 500ms (2 ครั้งต่อวินาที)
    }
    
    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
    };
  }, [isWebcamActive, loading]);

  const uploadImage = async (selectedFile) => {
    if (!selectedFile) {
      alert("Please select an image");
      return;
    }

    setLoading(false);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict-image/?preview=1', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result);

      if (Array.isArray(result.preview_base64_list) && result.preview_base64_list.length > 0) {
        const mime = result.preview_mime || 'image/jpeg';
        const previews = result.preview_base64_list.map(b64 =>
          `data:${mime};base64,${b64}`
        );
        setPreviewList(previews);
        setPreview(previews[0]);
      } else if (result.preview_base64) {
        const mime = result.preview_mime || 'image/jpeg';
        const previewUrl = `data:${mime};base64,${result.preview_base64}`;
        setPreview(previewUrl);
        setPreviewList([previewUrl]);
      }

      if (Array.isArray(result.faces) && result.faces.length > 0) {
        setPanDeg(result.faces[0].pan ?? 0);
      } else {
        setPanDeg(0);
      }

    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  const imgRef = useRef<HTMLImageElement>(null);
  const offsetX = 250;

  return (
    <div className='container-3'>
      <div className='use-ftf2'>
        <div className='wrapper'>
          <div className='box2'>
            <div className="button2-wrapper">
              <div className='button-2' onClick={handleImportClick}>
                Import Image
                <img src="/img/add-image.png" className='add_img' alt="Import" />
                <input
                  type="file"
                  id="imageInput"
                  ref={fileInputRef}
                  accept='image/*'
                  onChange={handleChange}
                  style={{ display: 'none' }}
                />
              </div>
            </div>

            <div className='button-3' onClick={handleVideoClick}>
              Import Video
              <img src="/img/video-production.png" className='add_img' alt="Video" />
              <input
                type="file"
                ref={videoInputRef}
                accept="video/*"
                onChange={handleVideoChange}
                style={{ display: 'none' }}
              />
            </div>

            <div 
              className='button-4' 
              onClick={handleWebcamClick}
              style={{ 
                backgroundColor: isWebcamActive ? '#4CAF50' : '',
                border: isWebcamActive ? '2px solid #45a049' : ''
              }}
            >
              {isWebcamActive ? '⏹ Stop Webcam' : 'Webcam'}
              <img src="/img/webcam.png" className='add_img' alt="Webcam" />
            </div>
          </div>

          <div className='box3'>
            {isWebcamActive ? (
              <div style={{ position: 'relative', width: '100%' }}>
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  videoConstraints={{
                    width: 1280,
                    height: 720,
                    facingMode: "user"
                  }}
                  style={{ 
                    width: '100%', 
                    height: 'auto',
                    borderRadius: '8px'
                  }}
                />
                <div style={{
                  position: 'absolute',
                  top: '10px',
                  right: '10px',
                  padding: '8px 16px',
                  color: 'white',
                  borderRadius: '20px',
                  fontSize: '12px',
                  fontWeight: 'bold'
                }}>
                </div>
              </div>
            ) : (
              <>
                {file && (
                  file.includes('video') || file.includes('.mp4') || file.includes('.mov') || file.includes('.avi') ?
                    <video src={file} controls style={{ maxWidth: '100%', height: 'auto' }} /> :
                    <img src={file} alt="Uploaded preview" style={{ display: 'none' }} />
                )}
              </>
            )}

            {preview && prediction && (
              <div className='preview-container'>
                <div className="preview-frame">
                  <div className="preview-header">
                    <span className="preview-title">
                      Detected {prediction.faces.length} Face{prediction.faces.length > 1 ? 's' : ''} :
                      Frontal = {prediction.faces.filter(f => f.pose === "Frontal").length},
                      Left = {prediction.faces.filter(f => f.pose === "Left Profile").length},
                      Right = {prediction.faces.filter(f => f.pose === "Right Profile").length}
                    </span>
                  </div>

                  <div className="preview-frame" style={{ position: 'relative' }}>
                    <img ref={imgRef} src={preview} className="preview-image" alt="Preview" />

                    {prediction.faces.map((face, index) => {
                      if (!face?.box) return null;
                      const { x, y, w, h } = face.box;

                      const renderedWidth = imgRef.current?.clientWidth || prediction.image_width;
                      const renderedHeight = imgRef.current?.clientHeight || prediction.image_height;
                      const scaleX = renderedWidth / prediction.image_width;
                      const scaleY = renderedHeight / prediction.image_height;

                      return (
                        <div
                          key={index}
                          className="model-overlay"
                          style={{
                            position: 'absolute',
                            left: `${(x - 100) * scaleX}px`,  
                            top: `${(y - 80) * scaleY}px`, 
                            width: '170px',
                            height: '120px',
                            display: 'flex',
                            flexDirection: 'row',
                            backgroundColor: 'rgba(0, 0, 0, 0.1)',
                            color: 'black',
                            fontSize: '14px',
                            padding: '8px',
                            borderRadius: '8px',
                            whiteSpace: 'pre-line',
                          }}
                        >
                          {face.pan.toFixed(1)}°
                          <div style={{ width: '90px', height: '140px', marginLeft: '-20px', marginTop: '-20px' }}>
                            <Canvas
                              camera={{ position: [0, 2, 6], fov: 35 }}
                              style={{ background: 'transparent' }}
                            >
                              <ambientLight intensity={0.6} />
                              <directionalLight position={[2, 2, 2]} intensity={0.8} />
                              <Suspense fallback={null}>
                                <group
                                  rotation={[0, THREE.MathUtils.degToRad(face.pan ?? 0), 0]} 
                                  scale={0.6}
                                >
                                  <Face emotion={face.emotion || "neutral"} />
                                </group>
                              </Suspense>
                            </Canvas>
                          </div>
                          <img
                            src={
                              face.emotion === 'Neutral'
                                ? '/emotions_image/Neutral.png'
                                : face.emotion === 'Happy'
                                ? '/emotions_image/Happy.png'
                                : face.emotion === 'Sad'
                                ? '/emotions_image/Sad.png'
                                : face.emotion === 'Angry'
                                ? '/emotions_image/Angry.png'
                                : face.emotion === 'Disgusted'
                                ? '/emotions_image/Disgust.png'
                                : face.emotion === 'Fearful'
                                ? '/emotions_image/Fear.png'
                                : face.emotion === 'Surprised'
                                ? '/emotions_image/Surprise.png'
                                : '/emotions_image/Neutral.png'
                            }
                            alt={face.emotion}
                            style={{
                              width: '160%',           
                              height: 'auto',
                              marginLeft: '-50px',   
                              borderRadius: '6px',
                              objectFit: 'contain',    
                              opacity: 1,            
                            }}
                          />
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Core_gen;