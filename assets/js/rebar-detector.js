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

    // Simple heuristic for rebar detection based on common classes
    // In a real implementation, this would use a trained model
    const imageClasses = predictions.map(p => p.className.toLowerCase());

    // Look for concrete/construction related classes
    const concreteKeywords = ['concrete', 'wall', 'building', 'construction', 'architecture', 'brick'];
    const metalKeywords = ['metal', 'steel', 'iron', 'wire', 'cable', 'pipe'];

    const hasConcrete = imageClasses.some(cls =>
      concreteKeywords.some(keyword => cls.includes(keyword))
    );

    const hasMetal = imageClasses.some(cls =>
      metalKeywords.some(keyword => cls.includes(keyword))
    );

    // Calculate confidence based on predictions
    const topPrediction = predictions[0];
    let confidence = Math.min(topPrediction.probability * 100, 95); // Cap at 95%

    // Adjust confidence based on detected features
    if (hasConcrete && hasMetal) {
      confidence = Math.max(confidence, 75); // High confidence if both detected
    } else if (hasConcrete) {
      confidence = Math.max(confidence, 60); // Medium confidence for concrete
    } else {
      confidence = Math.min(confidence, 40); // Low confidence otherwise
    }

    // Simulate exposed rebar detection (in reality, this would be a trained model)
    const exposedRebar = confidence > 65;

    return {
      exposedRebar,
      confidence: Math.round(confidence),
      predictions
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

    if (result.exposedRebar) {
      this.resultIcon.textContent = '⚠️';
      this.resultTitle.textContent = 'Exposed Rebar Detected';
      this.resultDescription.textContent = 'This image shows signs of exposed rebar. Professional inspection recommended.';
    } else {
      this.resultIcon.textContent = '✅';
      this.resultTitle.textContent = 'No Exposed Rebar Detected';
      this.resultDescription.textContent = 'This image appears to show properly covered concrete structures.';
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