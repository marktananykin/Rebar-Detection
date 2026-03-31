from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import shutil
import os
from inference import detect_rebar
import os

app = FastAPI(title="Rebar Detection API", description="API for detecting exposed rebar in concrete structures")

# Get Roboflow API key from environment
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY', 'VNbzzXtSvbYr0vWCuokM')  # Use provided key as default

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Structural Rebar Inspection</title>
        <style>
            :root {
                --primary: #1d4ed8;
                --secondary: #64748b;
                --silver: #d4d4d8;
                --background: #f8fafc;
                --surface: #ffffff;
                --text: #0f172a;
                --radius: 16px;
                --shadow: 0 14px 30px rgba(15, 23, 42, 0.12);
            }
            body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%); min-height: 100vh; color: var(--text); margin: 0; }
            .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1rem 3rem; }
            .hero { background: rgba(59, 130, 246, 0.08); border:1px solid var(--silver); border-radius: var(--radius); box-shadow: var(--shadow); padding: 1.75rem; text-align:center; }
            .hero h1 { color: var(--primary); margin:0; font-size:clamp(2rem,3vw,3rem); }
            .hero p { color: #475569; margin: 0.4rem 0; }
            .upload-form, .result, .no-exposed, .exposed, .error { border-radius:var(--radius); }
            .upload-form { margin:1.5rem 0; background:var(--surface); border:1px solid var(--silver); box-shadow:var(--shadow); padding:1.2rem; }
            input[type=file] { padding:0.8rem; border:1px solid var(--silver); border-radius:8px; width:100%; }
            button { background:var(--primary); color:white; border:none; border-radius:10px; padding:0.75rem 1.2rem; cursor:pointer; font-weight:700; transition:transform .15s ease; }
            button:hover { transform:translateY(-1px); }
            .result { background:#eef2ff; border:1px solid #c7d2fe; color: var(--text); margin-top:1rem; padding:1rem; }
            .exposed { background:#fef2f2; border:1px solid #fecaca; }
            .no-exposed { background:#ecfdf5; border:1px solid #bbf7d0; }
            .error { background:#fff7ed; border:1px solid #ffd8a8; }
        </style>
    </head>
    <body>
        <h1>Rebar Detection System</h1>
        <p>Upload an image of a concrete wall, column, or beam to detect if there is exposed rebar.</p>
        <p><strong>Note:</strong> Make sure to train the model first using the training script before using this interface.</p>

        <form class="upload-form" action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Detect Rebar</button>
        </form>

        <div id="result"></div>

        <script>
            const form = document.querySelector('form');
            const resultDiv = document.getElementById('result');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(form);

                resultDiv.innerHTML = '<p>Processing...</p>';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (response.ok) {
                        let className = result.prediction.includes('Exposed') ? 'exposed' : 'no-exposed';
                        const thresholdPct = result.confidence_threshold ? (result.confidence_threshold * 100).toFixed(2) : 'N/A';
                        resultDiv.innerHTML = `
                            <div class="result ${className}">
                                <h3>Prediction: ${result.prediction}</h3>
                                <p>Confidence: ${(result.confidence * 100).toFixed(2)}%</p>
                                <p>Confidence Threshold: ${thresholdPct}%</p>
                                <p>Model: ${result.method || 'unknown'}</p>
                                <p>Status: ${result.status || 'unknown'}</p>
                            </div>
                        `;
                    } else {
                        resultDiv.innerHTML = `
                            <div class="result error">
                                <h3>Error</h3>
                                <p>${result.detail}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<div class="result error"><p>Error processing image. Please try again.</p></div>';
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict_rebar(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Try Roboflow first, fallback to local model
        result = detect_rebar(temp_path, roboflow_api_key=ROBOFLOW_API_KEY)
    except Exception as e:
        # If Roboflow fails, try local model
        try:
            if not os.path.exists('rebar_model.pth'):
                raise HTTPException(status_code=400, detail="Model not found. Please train the model first using train.py")
            result = detect_rebar(temp_path)
        except Exception as local_e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}, Local model also failed: {str(local_e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
