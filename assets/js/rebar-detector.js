// Rebar Detection JavaScript
class RebarDetector {
  constructor() {
    this.model = null;
    this.isModelLoaded = false;
    this.roboflowApiKey = window.ROBOFLOW_API_KEY || '';
    this.roboflowWorkspace = window.ROBOFLOW_WORKSPACE || 'marks-workspace-dymtv';
    this.roboflowWorkflow = window.ROBOFLOW_WORKFLOW || 'general-segmentation-api';
    this.roboflowClasses = window.ROBOFLOW_CLASSES || 'Exposed rebar';

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
    // Convert image element to blob
    const response = await fetch(imageElement.src);
    const blob = await response.blob();

    // Create form data for the workflow
    const formData = new FormData();
    formData.append('file', blob);

    // Use Roboflow InferenceHTTPClient workflow approach
    const workflowUrl = `https://serverless.roboflow.com/${this.roboflowWorkspace}/${this.roboflowWorkflow}`;

    const rfResponse = await fetch(workflowUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.roboflowApiKey}`
      },
      body: formData
    });

    if (!rfResponse.ok) {
      throw new Error(`Roboflow workflow error: ${rfResponse.status} ${rfResponse.statusText}`);
    }

    const result = await rfResponse.json();

    // Parse workflow results for exposed rebar detection
    let hasExposedRebar = false;
    let maxConfidence = 0;
    let reasoning = '';

    // Check for segmentation results or object detection results
    if (result && result.result) {
      const workflowResult = result.result;

      // Look for exposed rebar in various result formats
      if (workflowResult.predictions) {
        // Object detection format
        hasExposedRebar = workflowResult.predictions.some(pred =>
          pred.class && pred.class.toLowerCase().includes('exposed') ||
          pred.class && pred.class.toLowerCase().includes('rebar')
        );

        maxConfidence = Math.max(...workflowResult.predictions.map(pred => pred.confidence || 0));
        reasoning = hasExposedRebar ?
          'Roboflow workflow detected exposed rebar in the image' :
          'Roboflow workflow analysis shows no exposed rebar detected';
      }
      else if (workflowResult.segmentation) {
        // Segmentation format
        hasExposedRebar = workflowResult.segmentation.some(seg =>
          seg.class && seg.class.toLowerCase().includes('exposed') ||
          seg.class && seg.class.toLowerCase().includes('rebar')
        );

        maxConfidence = workflowResult.segmentation.length > 0 ?
          Math.max(...workflowResult.segmentation.map(seg => seg.confidence || 0)) : 0;

        reasoning = hasExposedRebar ?
          'Roboflow segmentation detected exposed rebar regions' :
          'Roboflow segmentation shows no exposed rebar regions';
      }
      else if (workflowResult.classes) {
        // Class-based results
        hasExposedRebar = workflowResult.classes.some(cls =>
          cls.name && cls.name.toLowerCase().includes('exposed') ||
          cls.name && cls.name.toLowerCase().includes('rebar')
        );

        maxConfidence = workflowResult.classes.length > 0 ?
          Math.max(...workflowResult.classes.map(cls => cls.confidence || 0)) : 0;

        reasoning = hasExposedRebar ?
          'Roboflow classification detected exposed rebar' :
          'Roboflow classification shows no exposed rebar';
      }
    }

    // Fallback: check if the specified classes were detected
    if (!hasExposedRebar && result && result[this.roboflowClasses.toLowerCase()]) {
      hasExposedRebar = true;
      maxConfidence = result[this.roboflowClasses.toLowerCase()].confidence || 0.8;
      reasoning = `Roboflow detected ${this.roboflowClasses} in the image`;
    }

    const confidence = Math.min(100, Math.max(20, Math.round(maxConfidence * 100)));

    return {
      hasExposedRebar,
      confidence,
      reasoning
    };
  }

  async detectRebarWithMobileNet(imageElement) {
    const predictions = await this.model.classify(imageElement);
    const imageClasses = predictions.map(p => p.className.toLowerCase());
    const probabilities = predictions.map(p => p.probability);

    console.log('MobileNet predictions:', predictions);

    // Enhanced keyword sets for better rebar detection
    const concreteKeywords = [
      'concrete', 'wall', 'building', 'construction', 'architecture', 'brick', 'cement',
      'masonry', 'stone', 'plaster', 'stucco', 'mortar', 'beam', 'column', 'slab'
    ];

    const metalKeywords = [
      'metal', 'steel', 'iron', 'wire', 'cable', 'pipe', 'rebar', 'reinforcement',
      'bar', 'rod', 'girder', 'beam', 'structural', 'framework'
    ];

    const corrosionKeywords = [
      'rust', 'corrosion', 'oxidized', 'weathered', 'rusted', 'deteriorated',
      'worn', 'aged', 'patina', 'tarnished'
    ];

    const constructionKeywords = [
      'industrial', 'factory', 'warehouse', 'bridge', 'highway', 'infrastructure',
      'engineering', 'civil', 'structural'
    ];

    // Analyze predictions
    const hasConcrete = imageClasses.some(cls =>
      concreteKeywords.some(keyword => cls.includes(keyword))
    );
    const hasMetal = imageClasses.some(cls =>
      metalKeywords.some(keyword => cls.includes(keyword))
    );
    const hasCorrosion = imageClasses.some(cls =>
      corrosionKeywords.some(keyword => cls.includes(keyword))
    );
    const hasConstruction = imageClasses.some(cls =>
      constructionKeywords.some(keyword => cls.includes(keyword))
    );

    // Get confidence scores for relevant classes
    const concreteScore = Math.max(...imageClasses.map((cls, idx) =>
      concreteKeywords.some(keyword => cls.includes(keyword)) ? probabilities[idx] : 0
    ));

    const metalScore = Math.max(...imageClasses.map((cls, idx) =>
      metalKeywords.some(keyword => cls.includes(keyword)) ? probabilities[idx] : 0
    ));

    const corrosionScore = Math.max(...imageClasses.map((cls, idx) =>
      corrosionKeywords.some(keyword => cls.includes(keyword)) ? probabilities[idx] : 0
    ));

    // Advanced detection logic
    let confidence = 0;
    let hasExposedRebar = false;
    let reasoning = '';

    // High confidence: Corrosion + Concrete (strongest indicator of exposed rebar)
    if (hasCorrosion && hasConcrete && corrosionScore > 0.1) {
      confidence = Math.min(95, 70 + (corrosionScore * 100) * 0.3);
      hasExposedRebar = true;
      reasoning = 'Detected corrosion on concrete surface - strong indicator of exposed rebar';
    }
    // Medium-high confidence: Metal + Concrete + Construction context
    else if (hasMetal && hasConcrete && (hasConstruction || metalScore > 0.2)) {
      confidence = Math.min(85, 60 + (metalScore * 100) * 0.4);
      hasExposedRebar = true;
      reasoning = 'Detected metal elements in concrete construction context';
    }
    // Medium confidence: Just metal in construction setting
    else if (hasMetal && hasConstruction) {
      confidence = Math.min(75, 50 + (metalScore * 100) * 0.5);
      hasExposedRebar = true;
      reasoning = 'Detected metal in construction/industrial setting';
    }
    // Low confidence: Only concrete detected
    else if (hasConcrete) {
      confidence = Math.max(20, concreteScore * 100 * 0.3);
      hasExposedRebar = false;
      reasoning = 'Concrete structure detected but no exposed rebar visible';
    }
    // Very low confidence: No relevant features
    else {
      confidence = 10;
      hasExposedRebar = false;
      reasoning = 'No concrete or construction elements detected';
    }

    // Boost confidence if top prediction is very confident
    const topProbability = Math.max(...probabilities);
    if (topProbability > 0.8) {
      confidence = Math.min(confidence + 10, 95);
    }

    // Additional check: if metal is detected with reasonable confidence
    if (hasMetal && metalScore > 0.15 && !hasExposedRebar) {
      confidence = Math.min(confidence + 15, 80);
      hasExposedRebar = true;
      reasoning = 'Metal detected - possible exposed rebar';
    }

    return {
      hasExposedRebar,
      confidence: Math.max(5, Math.min(Math.round(confidence), 95)),
      reasoning
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