import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

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

    axios.post('http://localhost:5000/check-sleep', {
      image: image
    })
    .then((response) => {
      console.log(response.data);
      setResult(response.data);
    })
    .catch((error) => {
      console.log(error);
    });
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
    <div>
      <video autoPlay={true} ref={videoRef} width="640" height="480" />
      <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />
      {result ? (
        <p>API call returned: {result}</p>
      ) : (
        <p>Loading...</p>
      )}    
    </div>
  );
}

export default WebcamCapture;
