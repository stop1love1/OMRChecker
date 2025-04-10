# Hướng Dẫn Sử Dụng OMRChecker API

## Cấu Trúc Thư Mục

API đã được điều chỉnh để lưu trữ và xử lý các tệp theo cấu trúc thư mục của OMRChecker gốc:

```
OMRChecker/
│
├── inputs/                 # Thư mục đầu vào, API sẽ lưu template và ảnh tại đây
│   ├── directory_name_1/   # Mỗi bộ OMR sẽ được lưu trong một thư mục riêng biệt
│   │   ├── template.json   # Tệp template cho bộ OMR
│   │   └── image1.jpg      # Các ảnh OMR được quét
│   │   └── image2.jpg      
│   │
│   └── directory_name_2/   # Một bộ OMR khác
│       ├── template.json
│       └── image3.jpg
│
├── outputs/                # Thư mục đầu ra, kết quả xử lý OMR sẽ được lưu ở đây
│   ├── directory_name_1/   # Kết quả tương ứng với thư mục đầu vào
│   │   └── Results_11AM.csv    # Tệp kết quả chính với dữ liệu OMR đã xử lý
│   │
│   └── directory_name_2/
│
└── api_server.py           # Máy chủ API
```

## Cách Chạy API

### 1. Cài đặt Thư Viện

```bash
pip install -r requirements.txt
```

### 2. Chạy Máy Chủ API

**Trên Windows:**
```bash
python run_api.py
# hoặc
start_api.bat
```

**Trên Linux/Mac:**
```bash
python3 run_api.py
# hoặc
./start_api.sh
```

Máy chủ API sẽ khởi động tại địa chỉ mặc định: http://localhost:5000

### 3. Truy Cập API

- **Giao Diện Web:** [http://localhost:5000/](http://localhost:5000/)
- **Swagger UI Nâng Cao:** [http://localhost:5000/swagger](http://localhost:5000/swagger)
- **Tài Liệu Flask-RestX:** [http://localhost:5000/api/docs](http://localhost:5000/api/docs)

## API Endpoints

### 1. Xử Lý OMR Sheet

**Endpoint:** `POST /api/process-omr`

**Mô tả:** Upload tệp template JSON và ảnh OMR để xử lý. API sẽ lưu các tệp này vào thư mục `inputs/{directory_name}` và trả về nội dung của Results_11AM.csv.

**Parameters:**
- `template_file` (required): Tệp JSON định nghĩa bố cục OMR
- `image_file` (required): Tệp ảnh OMR (định dạng PNG, JPG, JPEG)
- `directory_name` (required): Tên thư mục sẽ được tạo trong thư mục inputs
- `include_images` (optional, default: false): Có kèm theo hình ảnh đã xử lý dưới dạng base64 hay không

**Ví dụ sử dụng curl:**
```bash
curl -X POST "http://localhost:5000/api/process-omr" \
  -H "Content-Type: multipart/form-data" \
  -F "template_file=@/path/to/template.json" \
  -F "image_file=@/path/to/omr_image.jpg" \
  -F "directory_name=my_omr_test"
```

**Phản hồi:**
```json
{
  "message": "OMR processing completed successfully",
  "result_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "input_dir": "inputs/my_omr_test",
  "output_dir": "outputs/my_omr_test",
  "csv_file": "Results_11AM.csv",
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

### 2. Lấy Kết Quả

**Endpoint:** `GET /api/results/{result_id}`

**Mô tả:** Lấy kết quả cho một OMR sheet đã xử lý trước đó. API sẽ ưu tiên đọc file Results_11AM.csv nếu có.

**Parameters:**
- `result_id`: ID của bộ kết quả (nhận được từ endpoint process-omr)

**Ví dụ sử dụng curl:**
```bash
curl -X GET "http://localhost:5000/api/results/f47ac10b-58cc-4372-a567-0e02b2c3d479"
```

**Phản hồi:**
```json
{
  "result_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "csv_file": "Results_11AM.csv",
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

### 3. Tải Xuống Tệp

**Endpoint:** `GET /api/download/{result_id}/{filename}`

**Mô tả:** Tải xuống một tệp (CSV, hình ảnh, v.v.) từ bộ kết quả. Nếu filename là "results.csv" hoặc "Results_11AM.csv", API sẽ ưu tiên trả về file Results_11AM.csv.

**Parameters:**
- `result_id`: ID của bộ kết quả
- `filename`: Đường dẫn của tệp cần tải xuống

**Ví dụ sử dụng curl:**
```bash
curl -X GET "http://localhost:5000/api/download/f47ac10b-58cc-4372-a567-0e02b2c3d479/Results_11AM.csv" --output results.csv
```

## Định Dạng Template.json

Template JSON cần tuân thủ cấu trúc của OMRChecker. Dưới đây là một mẫu đơn giản:

```json
{
  "pageDimensions": {
    "width": 1654,
    "height": 2339
  },
  "bubbleDimensions": {
    "width": 20,
    "height": 20
  },
  "preProcessors": [
    {
      "name": "CropPage",
      "options": {
        "relativePath": "crop_coordinates.json",
        "morphologySize": 5
      }
    }
  ],
  "fieldBlocks": {
    "Roll": {
      "fieldType": "QTYPE_ROLL",
      "origin": {"x": 700, "y": 400},
      "fieldLabels": ["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
      "directions": {"horizontal": 7, "vertical": 10},
      "fieldArea": {"x": 35, "y": 35},
      "options": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9}
    },
    "Q": {
      "fieldType": "QTYPE_MCQ",
      "origin": {"x": 800, "y": 650},
      "fieldLabels": ["Q1", "Q2", "Q3", "Q4", "Q5"],
      "directions": {"horizontal": 5, "vertical": 1},
      "fieldArea": {"x": 60, "y": 30},
      "options": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    }
  },
  "customLabels": {},
  "outputColumns": [],
  "emptyValue": "0",
  "options": {}
}
```

## Ví Dụ Tích Hợp

### Ví Dụ Python

```python
import requests
import os

# Đường dẫn đến template và ảnh
template_path = "path/to/template.json"
image_path = "path/to/image.jpg"
directory_name = "test_omr_batch"

# Gửi yêu cầu xử lý OMR
url = "http://localhost:5000/api/process-omr"
files = {
    'template_file': open(template_path, 'rb'),
    'image_file': open(image_path, 'rb')
}
data = {
    'directory_name': directory_name,
    'include_images': True
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Kết quả xử lý OMR:")
print(f"- ID kết quả: {result['result_id']}")
print(f"- Thư mục đầu vào: {result['input_dir']}")
print(f"- Thư mục đầu ra: {result['output_dir']}")
print(f"- Tệp CSV: {result['csv_file']}")
print(f"- Kết quả: {result['results']}")

# Tải xuống tệp Results_11AM.csv
if 'result_id' in result:
    csv_url = f"http://localhost:5000/api/download/{result['result_id']}/Results_11AM.csv"
    csv_response = requests.get(csv_url)
    with open('downloaded_results.csv', 'wb') as f:
        f.write(csv_response.content)
    print(f"Đã tải xuống kết quả vào tệp downloaded_results.csv")
```

## Lưu ý

1. API đã được điều chỉnh để lưu trữ template.json và ảnh vào thư mục `inputs/{directory_name}`, phù hợp với cấu trúc thư mục của OMRChecker.

2. Kết quả xử lý sẽ được lưu vào thư mục `outputs/{directory_name}` và API sẽ ưu tiên đọc file Results_11AM.csv (nếu có) để trả về kết quả.

3. Trong các yêu cầu tải xuống, nếu bạn chỉ định "results.csv" hoặc "Results_11AM.csv", API sẽ tự động tìm và trả về file Results_11AM.csv nếu có.

4. Các tệp đã tải lên sẽ vẫn còn trong thư mục `inputs`, bạn có thể chạy lại quá trình xử lý trực tiếp với OMRChecker bằng cách sử dụng lệnh:
   ```
   python main.py -i inputs/{directory_name}
   ``` 