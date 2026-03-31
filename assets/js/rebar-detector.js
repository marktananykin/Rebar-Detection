// Rebar Detection JavaScript using ONNX Runtime + Roboflow with robust upload handling
class RebarDetector {
  constructor() {
    this.session = null;
    this.isModelLoaded = false;
    this.modelPath = '/models/rebar_model.onnx';

    this.roboflowApiKey = window.ROBOFLOW_API_KEY || '';
    this.roboflowWorkspace = window.ROBOFLOW_WORKSPACE || 'marks-workspace-dymtv';
    this.roboflowWorkflow = window.ROBOFLOW_WORKFLOW || 'general-segmentation-api';
    this.roboflowEndpoint = `https://serverless.roboflow.com/${this.roboflowWorkspace}/${this.roboflowWorkflow}`;

    this.selectedFile = null;
    this.fallbackMode = false;

    this.initializeElements();
    this.setupEventListeners();
    this.loadModel();
  }

  async loadModel() {
    try {
      if (typeof ort !== 'undefined') {
        console.log('Loading YOLOv8 ONNX model...');
        this.session = await ort.InferenceSession.create(this.modelPath);
        this.isModelLoaded = true;
        console.log('Model loaded successfully');
      } else {
        console.warn('ONNX Runtime not available, skipping local model load.');
        this.fallbackMode = true;
      }
    } catch (error) {
      console.error('Failed to load ONNX model:', error);
      this.fallbackMode = true;
    }
  }

  initializeElements() {
    this.uploadArea = document.getElementById('uploadArea');
    this.fileInput = document.getElementById('fileInput');
    this.previewSection = document.getElementById('previewSection');
    this.previewImage = document.getElementById('previewImage');
    this.resultsSection = document.getElementById('resultsSection');
    this.resultIcon = document.getElementById('resultIcon');
    this.resultTitle = document.getElementById('resultTitle');
    this.resultDescription = document.getElementById('resultDescription');
    this.confidenceFill = document.getElementById('confidenceFill');
    this.confidencePercentage = document.getElementById('confidencePercentage');
    this.analyzeBtn = document.getElementById('analyzeBtn');
  }

  setupEventListeners() {
    this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

    this.uploadArea.addEventListener('click', () => this.fileInput.click());

    this.uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.uploadArea.style.borderColor = '#2563eb';
    });

    this.uploadArea.addEventListener('dragleave', (e) => {
      e.preventDefault();
      this.uploadArea.style.borderColor = '#e2e8f0';
    });

    this.uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      this.uploadArea.style.borderColor = '#e2e8f0';
      if (e.dataTransfer?.files?.length > 0) {
        this.handleFile(e.dataTransfer.files[0]);
      }
    });

    this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
  }

  handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
      this.handleFile(file);
    }
  }

  handleFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file.');
      return;
    }

    this.selectedFile = file;
    const reader = new FileReader();
    reader.onload = (event) => {
      this.previewImage.src = event.target.result;
      this.previewSection.style.display = 'block';
      this.uploadArea.style.display = 'none';
      this.resultsSection.style.display = 'none';
      this.resultTitle.textContent = 'Ready to analyze';
      this.resultDescription.textContent = 'Click Analyze to detect exposed rebar.';
      this.confidenceFill.style.width = '0%';
      this.confidencePercentage.textContent = '0%';
    };
    reader.readAsDataURL(file);
  }

  async analyzeImage() {
    if (!this.selectedFile) {
      alert('Please upload an image first.');
      return;
    }

    this.analyzeBtn.disabled = true;
    this.analyzeBtn.textContent = '🔄 Analyzing...';

    try {
      let result = null;

      if (this.roboflowApiKey) {
        result = await this.detectWithRoboflow(this.selectedFile);
      }

      if (!result || !result.hasExposedRebar && result.confidence < 0.35) {
        if (this.isModelLoaded && !this.fallbackMode) {
          result = await this.detectWithONNX(this.selectedFile);
        }
      }

      if (!result) {
        result = this.basicDetection();
      }

      this.displayResult(result);
    } catch (error) {
      console.error('Analysis failed:', error);
      this.displayError();
    } finally {
      this.analyzeBtn.disabled = false;
      this.analyzeBtn.textContent = '🔍 Analyze Image';
    }
  }

  async detectWithONNX(file) {
    if (!this.session || !this.isModelLoaded) {
      return {
        hasExposedRebar: false,
        confidence: 0,
        method: 'ONNX model unavailable'
      };
    }

    const img = new Image();
    img.src = URL.createObjectURL(file);

    await new Promise((r, rej) => {
      img.onload = r;
      img.onerror = rej;
    });

    const canvas = document.createElement('canvas');
    const size = 640;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    const scale = Math.max(size / img.width, size / img.height);
    const x = (size - img.width * scale) / 2;
    const y = (size - img.height * scale) / 2;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(img, x, y, img.width * scale, img.height * scale);

    const imageData = ctx.getImageData(0, 0, size, size);
    const { data } = imageData;
    const input = new Float32Array(1 * 3 * size * size);

    for (let i = 0; i < size * size; i += 1) {
      input[i] = data[i * 4] / 255.0;
      input[i + size * size] = data[i * 4 + 1] / 255.0;
      input[i + (2 * size * size)] = data[i * 4 + 2] / 255.0;
    }

    const tensor = new ort.Tensor('float32', input, [1, 3, size, size]);
    const outputs = await this.session.run({ images: tensor });
    const firstKey = Object.keys(outputs)[0];
    const outputTensor = outputs[firstKey];

    const outData = outputTensor.data;
    const outShape = outputTensor.dims;

    let maxConf = 0;
    let hasRebar = false;

    if (outShape.length === 3 && outShape[2] > 5) {
      const numDet = outShape[1];
      const boxes = outShape[2];

      for (let i = 0; i < numDet; i += 1) {
        const base = i * boxes;
        const objConf = outData[base + 4];
        let classIndex = 0;
        let classConf = 0;

        for (let c = 5; c < boxes; c += 1) {
          const score = outData[base + c];
          if (score > classConf) {
            classConf = score;
            classIndex = c - 5;
          }
        }

        const confidence = objConf * classConf;
        maxConf = Math.max(maxConf, confidence);

        if (classIndex === 0 && confidence >= 0.3) {
          hasRebar = true;
        }
      }
    }

    return {
      hasExposedRebar: hasRebar,
      confidence: maxConf,
      method: 'YOLOv8 ONNX'
    };
  }

  async detectWithRoboflow(file) {
    if (!this.roboflowApiKey) return null;

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(this.roboflowEndpoint, {
      method: 'POST',
      body: formData,
      headers: { 'Authorization': `Bearer ${this.roboflowApiKey}` }
    });

    if (!response.ok) {
      throw new Error(`Roboflow error ${response.status}`);
    }

    const data = await response.json();

    let predictions = [];
    if (data.result?.predictions) predictions = data.result.predictions;
    else if (data.result?.segmentation) predictions = data.result.segmentation;
    else if (data.predictions) predictions = data.predictions;

    const rebarPredictions = predictions.filter((pred) => {
      const cls = ((pred.class || pred.label || pred.name || '') + '').toLowerCase();
      return cls.includes('exposed') || cls.includes('rebar');
    });

    const maxConf = rebarPredictions.reduce((curr, p) => Math.max(curr, p.confidence || 0), 0);
    const hasRebar = maxConf >= 0.35;

    return {
      hasExposedRebar: hasRebar,
      confidence: maxConf,
      method: 'Roboflow API'
    };
  }

  basicDetection() {
    return {
      hasExposedRebar: false,
      confidence: 0.1,
      method: 'Fallback'
    };
  }

  displayResult(result) {
    this.resultsSection.style.display = 'block';

    const { hasExposedRebar, confidence, method } = result;
    const pct = Math.round(Math.min(Math.max(confidence * 100, 0), 100));

    if (hasExposedRebar) {
      this.resultIcon.textContent = '⚠️';
      this.resultTitle.textContent = 'Exposed Rebar Detected';
      this.resultDescription.textContent = `Detected by ${method}. Please inspect the area physically.`;
    } else {
      this.resultIcon.textContent = '✅';
      this.resultTitle.textContent = 'No Exposed Rebar Detected';
      this.resultDescription.textContent = `Analyzed by ${method}. Structure appears safe.`;
    }

    this.confidenceFill.style.width = `${pct}%`;
    this.confidencePercentage.textContent = `${pct}%`;

    if (pct >= 75) this.confidenceFill.style.backgroundColor = '#10b981';
    else if (pct >= 50) this.confidenceFill.style.backgroundColor = '#f59e0b';
    else this.confidenceFill.style.backgroundColor = '#ef4444';

    // ensure preview remains shown
    this.previewSection.style.display = 'block';
  }

  displayError() {
    this.resultsSection.style.display = 'block';
    this.resultIcon.textContent = '❌';
    this.resultTitle.textContent = 'Analysis Failed';
    this.resultDescription.textContent = 'Unable to analyze the image. Please try again.';
    this.confidenceFill.style.width = '0%';
    this.confidencePercentage.textContent = '0%';
  }
}

// Initialize detector when page is ready
window.addEventListener('DOMContentLoaded', () => {
  new RebarDetector();
});

function resetUpload() {
  window.location.reload();
}
