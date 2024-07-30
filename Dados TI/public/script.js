async function loadModels() {
    await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
    await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
    await faceapi.nets.faceRecognitionNet.loadFromUri('/models');
  }
  
  async function startVideo() {
    const video = document.getElementById('video');
    navigator.getUserMedia(
      { video: {} },
      stream => video.srcObject = stream,
      err => console.error(err)
    );
  }
  
  function fetchFaces() {
    return fetch('/faces')
      .then(response => response.json())
      .then(data => data);
  }
  
  async function recognizeFace(video, labeledDescriptors) {
    const options = new faceapi.TinyFaceDetectorOptions();
    const detections = await faceapi.detectAllFaces(video, options).withFaceLandmarks().withFaceDescriptors();
    
    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors);
    
    const results = detections.map(d => faceMatcher.findBestMatch(d.descriptor));
    results.forEach(result => {
      console.log(result.toString());
    });
  }
  
  document.addEventListener('DOMContentLoaded', async () => {
    await loadModels();
    startVideo();
  
    const video = document.getElementById('video');
    const labeledDescriptors = await fetchFaces().then(faces => {
      return Promise.all(faces.map(async face => {
        const img = await faceapi.fetchImage(`/Fotos/${face.id_rede}.jpg`);
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
        return new faceapi.LabeledFaceDescriptors(face.nome, [detections.descriptor]);
      }));
    });
  
    video.addEventListener('play', () => {
      setInterval(async () => {
        await recognizeFace(video, labeledDescriptors);
      }, 1000);
    });
  });
  