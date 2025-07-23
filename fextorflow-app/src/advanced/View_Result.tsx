import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './View_Result.css'

const ViewResult = (props) => {
  const [selectedFormat, setSelectedFormat] = useState('JSON');
  const [currentPage, setCurrentPage] = useState(1);
  
  const sampleData = {
    "status": "success",
    "processing_time": 1.23,
    "timestamp": "2024-07-23T10:30:45Z",
    "request_id": "req_abc123xyz",
    "face_detection": {
      "faces_found": 1,
      "confidence": 0.985,
      "bounding_box": {
        "x": 145,
        "y": 78,
        "width": 180,
        "height": 220
      }
    },
    "face_quality": "high",
    "angle": "frontal",
    "emotion_analysis": {
      "primary_emotion": {
        "label": "happy",
        "confidence": 0.873
      },
      "secondary_emotions": [
        {
          "label": "neutral",
          "confidence": 0.127
        }
      ]
    },
    "age_estimation": {
      "estimated_age": 28,
      "age_range": "25-32",
      "confidence": 0.91
    },
    "gender_detection": {
      "predicted_gender": "female",
      "confidence": 0.94
    }
  };

  const formatOptions = ['JSON', 'GRAPH', 'GRAPH'];

  const handleDownload = () => {
    const dataStr = JSON.stringify(sampleData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `data.${selectedFormat.toLowerCase()}`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const navigate = useNavigate();

  const handleHome = () => {
        navigate('/');
        props.setTrigger(false);
    };

  const handleBack = () => {
    navigate('/advanced/Model_1');
    props.setTrigger(false);
  };


  return (
    <div className="json-container">
      <div className="json-header">
        <div className="logo-v" onClick={handleHome}>
          <div className="fextorflow-v">
            Fextorflow
          </div>
        </div>
        <div className="json-nav">
          <span className="json-nav-item" onClick={handleHome}>HOME</span>
          <span className="json-nav-item" onClick={handleBack}>BACK</span>
        </div>
      </div>
      <div className="format-selector">
        <select 
          value={selectedFormat} 
          onChange={(e) => setSelectedFormat(e.target.value)}
          className="dropdown"
        >
          {formatOptions.map(format => (
            <option key={format} value={format}>{format}</option>
          ))}
        </select>
      </div>
      <div className="json-viewer-container">
        <div className="json-viewer">
          <pre className="json-content">
            {JSON.stringify(sampleData, null, 2)}
          </pre>
        </div>
        <div className="json-scrollbar">
          <div className="json-scroll-thumb"></div>
        </div>
      </div>
      <button className="json-download-button" onClick={handleDownload}>
        DOWNLOAD
      </button>
    </div>
  );
};

export default ViewResult;