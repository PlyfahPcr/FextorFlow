import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useEffect, useRef, useState } from 'react';
import './Compare.css';

interface CompareProps {
  setTrigger: (value: boolean) => void;
}

function Compare(props: CompareProps) {
  const [selectedFormat, setSelectedFormat] = useState('CK+');
  const [currentPage, setCurrentPage] = useState(1);
  const [popupImage, setPopupImage] = useState<string | null>(null);
  const [popupTitle, setPopupTitle] = useState<string>('');
  
  const formatOptions = ['CK+', 'JEFFE', 'RADVESS'];
  
  const getModelsForDataset = (dataset: string) => {
    return [
      {
        name: 'Model 1',
        image: `/public/img/${dataset.toLowerCase()}-model1.png`
      },
      {
        name: 'Model 2', 
        image: `/public/img/${dataset.toLowerCase()}-model2.png`
      },
      {
        name: 'Model 3',
        image: `/public/img/${dataset.toLowerCase()}-model3.png`
      }
    ];
  };
  
  const navigate = useNavigate();

  const handleHome = () => {
    navigate('/');
    props.setTrigger(false);
  };

  const handleBack = () => {
    navigate('/advanced');
    props.setTrigger(false);
  };

  const handleImageClick = (imageSrc: string, modelName: string) => {
    setPopupImage(imageSrc);
    setPopupTitle(modelName);
  };

  const closePopup = () => {
    setPopupImage(null);
    setPopupTitle('');
  };

  // Close popup when clicking outside or pressing Escape
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closePopup();
      }
    };

    if (popupImage) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden'; // Prevent background scrolling
    } else {
      document.body.style.overflow = 'unset';
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [popupImage]);

  return (
    <div className="container-c">
      <div className="header-c">
        <div className="logo-c">         
          <div className="fextorflow-c" onClick={handleHome}>
            Fextorflow
          </div> 
        </div>
        
        <div className="c-nav">
          <span className="c-nav-item" onClick={handleHome}>HOME</span>
          <span className="c-nav-item" onClick={handleBack}>MODEL</span>
        </div>          
      </div>
      <div className="content-c">
        <div className="selector-section-c">
          <span className="selector-label-c">Select Dataset:</span>
          <select 
            value={selectedFormat} 
            onChange={(e) => setSelectedFormat(e.target.value)}
            className="dropdown-c"
          >
            {formatOptions.map(format => (
              <option key={format} value={format}>{format}</option>
            ))}
          </select>
        </div>
        <div className="models-section-c">
          <div className="models-grid-c">
            {getModelsForDataset(selectedFormat).map((model, index) => (
              <div key={index} className="model-card-c">
                <h3 className="model-title-c">{model.name}</h3>
                <img 
                  src={model.image}
                  className="c-graph" 
                  alt={`${model.name} performance graph`}
                  onClick={() => handleImageClick(model.image, model.name)}
                  onError={(e) => {
                    (e.target as HTMLImageElement).src = '/public/img/model-graph.png';
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      {popupImage && (
        <div className="image-popup-overlay" onClick={closePopup}>
          <div className="image-popup-content" onClick={(e) => e.stopPropagation()}>
            <div className="image-popup-header">
              <h3 className="image-popup-title">{popupTitle}</h3>
              <button className="image-popup-close" onClick={closePopup}>
                ×
              </button>
            </div>
            <div className="image-popup-body">
              <img 
                src={popupImage} 
                alt={popupTitle}
                className="image-popup-img"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = '/public/img/model-graph.png';
                }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Compare;