// Rebar Detection JavaScript
class RebarDetector {
  constructor() {
    this.model = null;
    this.isModelLoaded = false;
    this.roboflowApiKey = window.ROBOFLOW_API_KEY || '';
    this.roboflowProject = window.ROBOFLOW_PROJECT || 'rebar-exposure-and-spalling/rebar-exposure-qm02o';
    this.roboflowModelVersion = window.ROBOFLOW_MODEL_VERSION || 1;

    this.initializeElements();
    this.setupEventListeners();
    this.loadModel();
  }

  initializeElements() {
    this.uploadArea = document.getElementById('uploadArea');
    this.fileInput = document.getElementById('fileInput');
    this.previewSection = document.getElementById('previewSection');
    this.previewImage = document.getElementById('previewImage');
    this.resultsSection = document.getElementById('resultsSection');
    this.resultCard = document.getElementById('resultCard');
    this.resultIcon = document.getElementById('resultIcon');
    this.resultTitle = document.getElementById('resultTitle');
    this.resultDescription = document.getElementById('resultDescription');
    this.confidenceFill = document.getElementById('confidenceFill');
    this.confidenceText = document.getElementById('confidenceText');
    this.analyzeBtn = document.getElementById('analyzeBtn');
  }

  setupEventListeners() {
    // File input change
    this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

    // Drag and drop
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
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        this.handleFile(files[0]);
      }
    });

    // Analyze button
    this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
  }

  async loadModel() {
    try {
      this.model = await mobilenet.load();
      this.isModelLoaded = true;
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
      this.showError('Failed to load AI model. Please refresh the page.');
    }
  }

  handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
      this.handleFile(file);
    }
  }

  handleFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      this.previewImage.src = e.target.result;
      this.uploadArea.style.display = 'none';
      this.previewSection.style.display = 'block';
    };
    reader.readAsDataURL(file);
  }

  async analyzeImage() {
    if (!this.isModelLoaded) {
      this.showError('AI model is still loading. Please wait.');
      return;
    }

    // Show loading state
    this.showLoading();

    try {
      // Get image element
      const img = this.previewImage;

      // Wait for image to load
      if (!img.complete) {
        await new Promise(resolve => {
          img.onload = resolve;
        });
      }

      // Analyze image
      const result = await this.detectRebar(img);

      // Show results
      this.showResult(result);

    } catch (error) {
      console.error('Analysis error:', error);
      this.showError('Failed to analyze image. Please try again.');
    }
  }

  async detectRebar(imageElement) {
    // Prefer Roboflow inference if API key is configured
    if (this.roboflowApiKey) {
      try {
        return await this.detectRebarWithRoboflow(imageElement);
      } catch (error) {
        console.warn('Roboflow inference error, falling back to local model:', error);
      }
    }

    // Local MobileNet heuristic fallback
    return await this.detectRebarWithMobileNet(imageElement);
  }

  async detectRebarWithRoboflow(imageElement) {
    const apiUrl = `https://detect.roboflow.com/${this.roboflowProject}/${this.roboflowModelVersion}`;

    // Convert image element to blob and send to Roboflow
    const response = await fetch(imageElement.src);
    const blob = await response.blob();

    const formData = new FormData();
    formData.append('file', blob);
    formData.append('api_key', this.roboflowApiKey);

    const rfResponse = await fetch(apiUrl, {
      method: 'POST',
      body: formData
    });

    if (!rfResponse.ok) {
      throw new Error(`Roboflow API error: ${rfResponse.status} ${rfResponse.statusText}`);
    }

    const data = await rfResponse.json();
    const containsExposure = data?.predictions?.some(pred => pred.class.toLowerCase().includes('rebar'));
    const hasSpall = data?.predictions?.some(pred => pred.class.toLowerCase().includes('spall'));

    const exposureProbability = data?.predictions?.reduce((acc, pred) => {
      const name = pred.class.toLowerCase();
      if (name.includes('rebar') || name.includes('exposure') || name.includes('spall')) {
        return Math.max(acc, pred.confidence || 0);
      }
      return acc;
    }, 0);

    const exposed = containsExposure || hasSpall;
    const confidence = Math.min(100, Math.max(20, Math.round(exposureProbability * 100)));

    return {
      hasExposedRebar: exposed,
      confidence,
      reasoning: exposed ? 'Roboflow object detection indicates exposed rebar/spalling' : 'Roboflow object detection indicates no visible exposed rebar'
    };
  }

  async detectRebarWithMobileNet(imageElement) {
    const predictions = await this.model.classify(imageElement);
    const imageClasses = predictions.map(p => p.className.toLowerCase());
    const probabilities = predictions.map(p => p.probability);

    console.log('MobileNet predictions:', predictions);

    const concreteKeywords = ['concrete', 'wall', 'building', 'construction', 'architecture', 'brick', 'cement', 'masonry'];
    const metalKeywords = ['metal', 'steel', 'iron', 'wire', 'cable', 'pipe', 'rebar', 'reinforcement'];
    const corrosionKeywords = ['rust', 'corrosion', 'oxidized', 'weathered'];

    const hasConcrete = imageClasses.some(cls => concreteKeywords.some(keyword => cls.includes(keyword)));
    const hasMetal = imageClasses.some(cls => metalKeywords.some(keyword => cls.includes(keyword)));
    const hasCorrosion = imageClasses.some(cls => corrosionKeywords.some(keyword => cls.includes(keyword)));

    let confidence = 0;
    let hasExposedRebar = false;

    if (hasMetal && hasConcrete) {
      confidence = 70;
      hasExposedRebar = true;
    } else if (hasCorrosion && hasConcrete) {
      confidence = 85;
      hasExposedRebar = true;
    } else if (hasConcrete) {
      confidence = 25;
      hasExposedRebar = false;
    } else {
      confidence = 10;
      hasExposedRebar = false;
    }

    const topProbability = probabilities[0] || 0;
    if (topProbability > 0.8) confidence = Math.min(confidence + 15, 95);
    else if (topProbability > 0.6) confidence = Math.min(confidence + 10, 90);

    const metalIndices = imageClasses.map((cls, idx) => (metalKeywords.some(keyword => cls.includes(keyword)) ? idx : -1)).filter(idx => idx !== -1);
    if (metalIndices.length > 0) {
      const metalProbability = Math.max(...metalIndices.map(idx => probabilities[idx]));
      if (metalProbability > 0.3) {
        confidence = Math.min(confidence + 20, 95);
        hasExposedRebar = true;
      }
    }

    return {
      hasExposedRebar,
      confidence: Math.max(5, Math.min(confidence, 95)),
      reasoning: hasExposedRebar ? 'Detected metal/corrosion signals in concrete structure' : 'No exposed rebar detected'
    };
  }

  showLoading() {
    this.resultsSection.style.display = 'block';
    this.resultIcon.textContent = '⏳';
    this.resultIcon.className = 'result-icon loading';
    this.resultTitle.textContent = 'Analyzing...';
    this.resultDescription.textContent = 'Please wait while we process your image.';
    this.confidenceFill.style.width = '0%';
    this.confidenceText.textContent = '0%';
  }

  showResult(result) {
    this.resultsSection.style.display = 'block';
    this.resultIcon.className = 'result-icon';

    if (result.hasExposedRebar) {
      this.resultIcon.textContent = '⚠️';
      this.resultTitle.textContent = 'Exposed Rebar Detected';
      this.resultDescription.textContent = result.reasoning + '. Professional inspection recommended.';
    } else {
      this.resultIcon.textContent = '✅';
      this.resultTitle.textContent = 'No Exposed Rebar Detected';
      this.resultDescription.textContent = result.reasoning + '. Structure appears intact.';
    }

    // Animate confidence bar
    this.confidenceFill.style.width = `${result.confidence}%`;
    this.confidenceText.textContent = `${result.confidence}%`;
  }

  showError(message) {
    this.resultsSection.style.display = 'block';
    this.resultIcon.textContent = '❌';
    this.resultIcon.className = 'result-icon';
    this.resultTitle.textContent = 'Analysis Failed';
    this.resultDescription.textContent = message;
    this.confidenceFill.style.width = '0%';
    this.confidenceText.textContent = '0%';
  }
}

// Global function for reset
function resetUpload() {
  document.getElementById('uploadArea').style.display = 'block';
  document.getElementById('previewSection').style.display = 'none';
  document.getElementById('resultsSection').style.display = 'none';
  document.getElementById('fileInput').value = '';
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new RebarDetector();
});