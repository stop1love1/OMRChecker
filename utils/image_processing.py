"""
Image processing utilities for OMRChecker API
"""
import os
import logging
import concurrent.futures
import time
from pathlib import Path
from src.logger import logger

def process_pdf_page(page_info):
    """Process a single PDF page with PyMuPDF"""
    page, matrix, output_path = page_info
    # Use JPEG output with quality compression for faster processing
    pix = page.get_pixmap(matrix=matrix)
    pix.save(output_path)
    # Release memory explicitly
    pix = None
    return output_path

def process_pdf2image_page(page_info):
    """Process a single page with pdf2image"""
    image, output_path = page_info
    # Use lower quality for improved performance
    image.save(output_path, 'JPEG', quality=70)
    return output_path

def process_pdf(pdf_path, input_dir, dpi=100, quality=70, max_workers=12):
    """
    Process PDF file and convert to images
    
    Args:
        pdf_path: Path to the PDF file
        input_dir: Directory where to save the converted images
        dpi: DPI for rendering (lower = faster but less detail)
        quality: JPEG quality (lower = smaller files, faster)
        max_workers: Maximum number of concurrent workers for processing
        
    Returns:
        List of paths to converted images
    """
    start_time = time.time()
    
    image_paths = []
    pdf_processed = False
    
    # Try PyMuPDF first (faster than pdf2image)
    try:
        import fitz
        
        # Use memory-optimized document opening
        pdf_document = fitz.open(pdf_path, filetype="pdf")
        
        if pdf_document.page_count == 0:
            logger.warning(f"PDF file has no pages: {os.path.basename(pdf_path)}")
            return image_paths
            
        # Calculate matrix based on provided DPI (default 72dpi in PDF)
        # Use lower DPI for faster processing
        matrix = fitz.Matrix(dpi/72, dpi/72)
        
        # Process pages in parallel for speed
        tasks = []
        for i in range(pdf_document.page_count):
            page = pdf_document[i]
            img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i+1}.jpg"
            img_path = os.path.join(input_dir, img_filename)
            tasks.append((page, matrix, img_path))
        
        # Use more workers and optimize thread pooling
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process in chunks for better memory management
            chunk_size = min(10, len(tasks))  # Process in chunks of 10 pages
            for i in range(0, len(tasks), chunk_size):
                chunk_tasks = tasks[i:i+chunk_size]
                futures = {executor.submit(process_pdf_page, task): task for task in chunk_tasks}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        path = future.result()
                        image_paths.append(path)
                    except Exception as e:
                        logger.error(f"Error processing PDF page: {str(e)}")
        
        pdf_document.close()
        pdf_processed = True
        
    except ImportError as e:
        logger.warning(f"PyMuPDF not available: {str(e)}. Will try pdf2image...")
    except Exception as e:
        logger.warning(f"Error using PyMuPDF: {str(e)}. Will try pdf2image...")
        
    # Fall back to pdf2image if PyMuPDF failed
    if not pdf_processed:
        try:
            from pdf2image import convert_from_path
            
            try:
                # Optimize pdf2image conversion with multithreading and lower DPI
                pdf_images = convert_from_path(
                    pdf_path, 
                    dpi=dpi,  # Lower DPI for faster processing
                    thread_count=max(1, max_workers//2),  # Use parallel processing but avoid too many threads
                    use_pdftocairo=True,  # Faster than pdftoppm
                    output_folder=input_dir,  # Write directly to output folder
                    fmt="jpeg",  # Use JPEG format - much faster than PNG
                    jpegopt={"quality": quality, "optimize": True, "progressive": False},
                    paths_only=True,  # Return paths only to reduce memory usage
                )
                
                # If paths_only works, we have the paths already
                if isinstance(pdf_images, list) and pdf_images and isinstance(pdf_images[0], str):
                    image_paths = pdf_images
                    pdf_processed = True
                else:
                    # Process images in parallel - only if paths_only didn't work
                    tasks = []
                    for i, image in enumerate(pdf_images):
                        img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i+1}.jpg"
                        img_path = os.path.join(input_dir, img_filename)
                        tasks.append((image, img_path))
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_image = {executor.submit(process_pdf2image_page, task): task for task in tasks}
                        for future in concurrent.futures.as_completed(future_to_image):
                            try:
                                path = future.result()
                                image_paths.append(path)
                            except Exception as e:
                                logger.error(f"Error saving PDF image: {str(e)}")
                    
                    pdf_processed = True
                
            except Exception as poppler_err:
                logger.warning(f"Default poppler path failed: {str(poppler_err)}. Trying alternate paths...")
                
                # Try different poppler paths for different systems
                poppler_paths = [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/homebrew/bin',  # MacOS Homebrew
                    '/usr/lib/x86_64-linux-gnu',
                    'C:/Program Files/poppler/bin',  # Windows paths
                    'C:/poppler/bin'
                ]
                
                for poppler_path in poppler_paths:
                    try:
                        pdf_images = convert_from_path(
                            pdf_path, 
                            dpi=dpi,
                            thread_count=max(1, max_workers//2),
                            use_pdftocairo=True,
                            poppler_path=poppler_path,
                            output_folder=input_dir,
                            fmt="jpeg",
                            jpegopt={"quality": quality, "optimize": True, "progressive": False},
                            paths_only=True
                        )
                        
                        if isinstance(pdf_images, list) and pdf_images and isinstance(pdf_images[0], str):
                            image_paths = pdf_images
                            pdf_processed = True
                            break
                        
                        # Process images in parallel - only if paths_only didn't work
                        tasks = []
                        for i, image in enumerate(pdf_images):
                            img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i+1}.jpg"
                            img_path = os.path.join(input_dir, img_filename)
                            tasks.append((image, img_path))
                        
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            future_to_image = {executor.submit(process_pdf2image_page, task): task for task in tasks}
                            for future in concurrent.futures.as_completed(future_to_image):
                                try:
                                    path = future.result()
                                    image_paths.append(path)
                                except Exception as e:
                                    logger.error(f"Error saving PDF image: {str(e)}")
                        
                        pdf_processed = True
                        break
                        
                    except Exception as path_err:
                        logger.warning(f"Failed with poppler path {poppler_path}: {str(path_err)}")
                else:
                    raise Exception("All poppler paths failed. Make sure poppler-utils is correctly installed.")
            
        except ImportError as e:
            logger.error(f"pdf2image not available: {str(e)}")
            raise ImportError("Neither PyMuPDF nor pdf2image are installed. Please install at least one of these libraries to process PDF files: 'pip install PyMuPDF pdf2image'")
        except Exception as e:
            logger.error(f"Error processing PDF with pdf2image: {str(e)}")
            raise
    
    if not pdf_processed:
        raise Exception("Failed to process PDF with any available method")
        
    end_time = time.time()
    processing_time = end_time - start_time
    page_count = len(image_paths)
    logger.info(f"PDF processing completed in {processing_time:.2f} seconds: {os.path.basename(pdf_path)}, {page_count} pages")
    
    return image_paths 