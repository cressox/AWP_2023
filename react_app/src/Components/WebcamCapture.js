import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import styles from './WebcamCapture.module.css';

const WebcamCapture = () => {
  // Erstelle Referenzen zu den Video- und Canvas-Elementen
  const videoRef = useRef();
  const canvasRef = useRef();
  const [result, setResult] = useState(null);

  // Beim ersten Laden der Komponente wird geprüft, ob Zugriff auf die
  // Kamera des Geräts möglich ist. Wenn ja, wird der Videostream
  // in das Video-Element geladen.
  useEffect(() => {
    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
          videoRef.current.srcObject = stream;
        })
        .catch((error) => {
          console.log('Error accessing video stream:', error);
        });
    }
  }, []);

  // Funktion zur Aufnahme eines Video-Frames. Das Bild wird in das
  // Canvas-Element gezeichnet und anschließend als Base64-String
  // im JPEG-Format abgespeichert. Dieses Bild wird an den Server
  // gesendet.
  const captureFrame = () => {
    canvasRef.current.getContext('2d').drawImage(videoRef.current, 0, 0, 640, 480);
    const image = canvasRef.current.toDataURL('image/jpeg', 1.0);

    const formData = new FormData();
    formData.append('image', dataURLtoBlob(image));

    axios.post('http://localhost:5000/check-face-slow', formData)
      .then((response) => {
        console.log(response.data);
        setResult(response.data.face_detected.toString());
      })    
      .catch((error) => {
        console.log(error);
      });

    // Hilfsfunktion, um das Base64-codierte Bild in ein Blob-Objekt zu konvertieren
    function dataURLtoBlob(dataURL) {
      const arr = dataURL.split(',');
      const mime = arr[0].match(/:(.*?);/)[1];
      const bstr = atob(arr[1]);
      let n = bstr.length;
      const u8arr = new Uint8Array(n);
      while(n--){
        u8arr[n] = bstr.charCodeAt(n);
      }
      return new Blob([u8arr], {type:mime});
    }
  }

  // Aufruf der captureFrame-Funktion alle 1000ms (1s).
  useEffect(() => {
    const interval = setInterval(() => {
      captureFrame();
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Rendert das Video- und das Canvas-Element
  return (
    <div className={result === 'true' ? styles['result-true'] : styles['result-false']}>
      <video autoPlay={true} ref={videoRef} width="640" height="480" />
      <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />
      {result ? (
        <p>API call returned:<h1>{result}</h1></p>
      ) : (
        <p>Loading...</p>
      )}
    </div>

  );
}

export default WebcamCapture;
