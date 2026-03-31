// Rebar Detection JavaScript
class RebarDetector {
  constructor() {
    this.model = null;
    this.isModelLoaded = false;
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
    // Use MobileNet to classify the image
    const predictions = await this.model.classify(imageElement);

    // Enhanced rebar detection logic
    const imageClasses = predictions.map(p => p.className.toLowerCase());
    const probabilities = predictions.map(p => p.probability);

    console.log('Predictions:', predictions); // Debug logging

    // Look for concrete/construction related classes
    const concreteKeywords = ['concrete', 'wall', 'building', 'construction', 'architecture', 'brick', 'cement', 'masonry'];
    const metalKeywords = ['metal', 'steel', 'iron', 'wire', 'cable', 'pipe', 'rebar', 'reinforcement'];
    const corrosionKeywords = ['rust', 'corrosion', 'oxidized', 'weathered'];

    const hasConcrete = imageClasses.some(cls =>
      concreteKeywords.some(keyword => cls.includes(keyword))
    );

    const hasMetal = imageClasses.some(cls =>
      metalKeywords.some(keyword => cls.includes(keyword))
    );

    const hasCorrosion = imageClasses.some(cls =>
      corrosionKeywords.some(keyword => cls.includes(keyword))
    );

    // Calculate confidence based on detected features
    let confidence = 0;
    let hasExposedRebar = false;

    // Primary indicators of exposed rebar
    if (hasMetal && hasConcrete) {
      // Metal + concrete suggests possible rebar exposure
      confidence = 70;
      hasExposedRebar = true;
    } else if (hasCorrosion && hasConcrete) {
      // Corrosion + concrete is a strong indicator
      confidence = 85;
      hasExposedRebar = true;
    } else if (hasConcrete) {
      // Just concrete - likely no exposed rebar
      confidence = 20;
      hasExposedRebar = false;
    } else {
      // No concrete detected - probably not a relevant image
      confidence = 10;
      hasExposedRebar = false;
    }

    // Adjust confidence based on top prediction probability
    const topProbability = probabilities[0];
    if (topProbability > 0.8) {
      confidence = Math.min(confidence + 15, 95);
    } else if (topProbability > 0.6) {
      confidence = Math.min(confidence + 10, 90);
    }

    // Special case: if we detect metal objects prominently, increase confidence
    const metalIndices = imageClasses.map((cls, idx) =>
      metalKeywords.some(keyword => cls.includes(keyword)) ? idx : -1
    ).filter(idx => idx !== -1);

    if (metalIndices.length > 0) {
      const metalProbability = Math.max(...metalIndices.map(idx => probabilities[idx]));
      if (metalProbability > 0.3) {
        confidence = Math.min(confidence + 20, 95);
        hasExposedRebar = true;
      }
    }

    return {
      hasExposedRebar: hasExposedRebar,
      confidence: Math.max(5, Math.min(confidence, 95)), // Clamp between 5-95%
      reasoning: hasExposedRebar ?
        `Detected ${hasCorrosion ? 'corroded ' : ''}metal elements in concrete structure` :
        'No exposed rebar detected in concrete structure'
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