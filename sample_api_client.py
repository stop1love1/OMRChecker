#!/usr/bin/env python3
"""
OMRChecker API Client Example

Script này minh họa cách sử dụng OMRChecker API từ client Python
"""

import os
import sys
import requests
import json
import argparse

def process_omr_sheet(api_url, template_file, image_file, directory_name, include_images=False):
    """
    Xử lý một tờ OMR bằng cách gửi yêu cầu đến API
    
    Parameters:
    - api_url: URL của API (ví dụ: http://localhost:5000)
    - template_file: Đường dẫn đến file template.json
    - image_file: Đường dẫn đến file ảnh OMR
    - directory_name: Tên thư mục sẽ được tạo trong inputs/
    - include_images: Có gồm hình ảnh đã xử lý trong kết quả hay không
    
    Returns:
    - Đối tượng JSON từ API response
    """
    url = f"{api_url}/api/process-omr"
    
    # Kiểm tra các tệp có tồn tại không
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Template file not found: {template_file}")
    
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file not found: {image_file}")
    
    # Mở và chuẩn bị các tệp để tải lên
    files = {
        'template_file': open(template_file, 'rb'),
        'image_file': open(image_file, 'rb')
    }
    
    data = {
        'directory_name': directory_name,
        'include_images': include_images
    }
    
    try:
        # Gửi yêu cầu POST đến API
        print(f"Sending request to {url}...")
        response = requests.post(url, files=files, data=data)
        
        # Kiểm tra lỗi
        response.raise_for_status()
        
        # Phân tích kết quả JSON
        result = response.json()
        return result
    
    except requests.exceptions.RequestException as e:
        if hasattr(e.response, 'text'):
            print(f"Error: {e.response.text}")
        else:
            print(f"Error: {str(e)}")
        return None
    
    finally:
        # Đóng các tệp đã mở
        for f in files.values():
            f.close()

def download_result_file(api_url, result_id, filename, output_path=None):
    """
    Tải xuống một tệp kết quả từ API
    
    Parameters:
    - api_url: URL của API
    - result_id: ID kết quả từ phản hồi xử lý OMR
    - filename: Tên tệp cần tải xuống
    - output_path: Đường dẫn để lưu tệp tải xuống (nếu None, sử dụng tên tệp gốc)
    
    Returns:
    - Đường dẫn đến tệp đã tải xuống
    """
    url = f"{api_url}/api/download/{result_id}/{filename}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Xác định đường dẫn đầu ra
        if output_path is None:
            output_path = filename
        
        # Lưu tệp
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return output_path
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {str(e)}")
        return None

def display_results(result):
    """Hiển thị kết quả OMR theo định dạng đẹp"""
    if not result:
        return
    
    print("\n" + "="*60)
    print("OMR PROCESSING RESULTS")
    print("="*60)
    
    print(f"Result ID: {result.get('result_id')}")
    print(f"Message: {result.get('message')}")
    print(f"Input directory: {result.get('input_dir')}")
    print(f"Output directory: {result.get('output_dir')}")
    print(f"CSV file: {result.get('csv_file', 'N/A')}")
    
    print("\nResults:")
    for item in result.get('results', []):
        print(f"  File: {item.get('file_name', 'Unknown')}")
        for key, value in item.items():
            if key != 'file_name':
                print(f"    {key}: {value}")
        print("  " + "-"*40)
    
    print("="*60)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='OMRChecker API Client Example')
    parser.add_argument('--api-url', default='http://localhost:5000', help='API URL (default: http://localhost:5000)')
    parser.add_argument('--template', default='sample_template.json', help='Path to template.json file')
    parser.add_argument('--image', required=True, help='Path to OMR image file')
    parser.add_argument('--dir-name', default='api_test', help='Directory name to create in inputs/')
    parser.add_argument('--include-images', action='store_true', help='Include processed images in result')
    parser.add_argument('--download-csv', action='store_true', help='Download Results_11AM.csv file')
    
    args = parser.parse_args()
    
    # Process OMR sheet
    result = process_omr_sheet(
        args.api_url,
        args.template,
        args.image,
        args.dir_name,
        args.include_images
    )
    
    # Display results
    if result:
        display_results(result)
        
        # Download CSV if requested
        if args.download_csv and 'result_id' in result:
            csv_path = download_result_file(args.api_url, result['result_id'], 'Results_11AM.csv', 'omr_results.csv')
            if csv_path:
                print(f"\nCSV results downloaded to: {csv_path}")

if __name__ == "__main__":
    main() 