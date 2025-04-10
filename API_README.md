# OMRChecker API

This is a REST API for the OMRChecker project, allowing you to process OMR sheets via HTTP requests.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the API server:
   ```
   python run_api.py
   ```

The server will start on port 5000 by default. You can access the API in multiple ways:
- Web Interface: [http://localhost:5000/](http://localhost:5000/)
- Enhanced Swagger UI: [http://localhost:5000/swagger](http://localhost:5000/swagger)
- Flask-RestX Docs: [http://localhost:5000/api/docs](http://localhost:5000/api/docs)

## API Documentation

The API is fully documented using Swagger, providing an interactive way to explore and test the endpoints. You can access it at `/swagger`.

### API Endpoints

All API endpoints are prefixed with `/api`.

#### 1. Process OMR Sheet

**Endpoint:** `POST /api/process-omr`

This endpoint allows you to upload a template JSON file and an OMR image for processing.

**Parameters:**
- `template_file` (required): JSON template file defining the OMR layout
- `image_file` (required): OMR image file (PNG, JPG, JPEG)
- `directory_name` (required): Name of the directory to create
- `include_images` (optional, default: false): Whether to include base64 encoded processed images in the response

**Example using curl:**
```bash
curl -X POST "http://localhost:5000/api/process-omr" \
  -H "Content-Type: multipart/form-data" \
  -F "template_file=@/path/to/template.json" \
  -F "image_file=@/path/to/omr_image.jpg" \
  -F "directory_name=my_omr_test"
```

**Response:**
```json
{
  "message": "OMR processing completed successfully",
  "result_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "results": [
    {
      "file_name": "omr_image.jpg",
      "q1": "A",
      "q2": "B",
      ...
    }
  ]
}
```

#### 2. Get Results

**Endpoint:** `GET /api/results/{result_id}`

Retrieve the results for a previously processed OMR sheet.

**Parameters:**
- `result_id`: The ID of the result set (returned from the process-omr endpoint)

**Example using curl:**
```bash
curl -X GET "http://localhost:5000/api/results/f47ac10b-58cc-4372-a567-0e02b2c3d479"
```

#### 3. Download File

**Endpoint:** `GET /api/download/{result_id}/{filename}`

Download a file (CSV, image, etc.) from the result set.

**Parameters:**
- `result_id`: The ID of the result set
- `filename`: The path of the file to download

**Example using curl:**
```bash
curl -X GET "http://localhost:5000/api/download/f47ac10b-58cc-4372-a567-0e02b2c3d479/results.csv" --output results.csv
```

#### 4. Health Check

**Endpoint:** `GET /api/health`

Check if the API server is running.

**Example using curl:**
```bash
curl -X GET "http://localhost:5000/api/health"
```

## Swagger Support

The API supports two different Swagger UIs:

1. **Enhanced Swagger UI** at `/swagger` - A more feature-rich Swagger UI with better visualization and testing capabilities
2. **Flask-RestX Docs** at `/api/docs` - The standard Flask-RestX Swagger interface

Both provide:
- Interactive documentation
- Request/response examples
- Try-it-out functionality
- Models and schema definitions

## Integration Examples

### Python Example

```python
import requests

# Process an OMR sheet
url = "http://localhost:5000/api/process-omr"
files = {
    'template_file': open('template.json', 'rb'),
    'image_file': open('omr_image.jpg', 'rb')
}
data = {
    'directory_name': 'test_omr'
}

response = requests.post(url, files=files, data=data)
result = response.json()
result_id = result['result_id']

# Download results
csv_url = f"http://localhost:5000/api/download/{result_id}/results.csv"
csv_response = requests.get(csv_url)
with open('downloaded_results.csv', 'wb') as f:
    f.write(csv_response.content)
```

### JavaScript Example

```javascript
// Using fetch API
async function processOMR() {
  const formData = new FormData();
  formData.append('template_file', document.getElementById('templateFile').files[0]);
  formData.append('image_file', document.getElementById('imageFile').files[0]);
  formData.append('directory_name', 'js_test');

  const response = await fetch('http://localhost:5000/api/process-omr', {
    method: 'POST',
    body: formData
  });

  const result = await response.json();
  console.log(result);
  
  // Download results
  if (result.result_id) {
    window.location.href = `http://localhost:5000/api/download/${result.result_id}/results.csv`;
  }
}
```

## Notes

- The processed results are stored temporarily on the server and may be cleaned up periodically.
- For production use, consider adding authentication and rate limiting.
- The API is versioned (v1.0) and follows RESTful principles. 