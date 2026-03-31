from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import shutil
import os
from inference import detect_rebar

app = FastAPI(title="Rebar Detection API", description="API for detecting exposed rebar in concrete structures")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rebar Detection</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-form { margin: 20px 0; }
            .result { margin: 20px 0; padding: 10px; border-radius: 5px; }
            .exposed { background-color: #ffebee; border: 1px solid #f44336; }
            .no-exposed { background-color: #e8f5e8; border: 1px solid #4caf50; }
            .error { background-color: #fff3e0; border: 1px solid #ff9800; }
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
                        resultDiv.innerHTML = `
                            <div class="result ${className}">
                                <h3>Prediction: ${result.prediction}</h3>
                                <p>Confidence: ${(result.confidence * 100).toFixed(2)}%</p>
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
    # Check if model file exists
    if not os.path.exists('rebar_model.pth'):
        raise HTTPException(status_code=400, detail="Model not found. Please train the model first using train.py")

    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Perform prediction
        result = detect_rebar(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
