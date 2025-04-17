"""
Entry file for OMRChecker with simplified timing logs
"""
import os
from csv import QUOTE_NONNUMERIC
from pathlib import Path
from time import time
import concurrent.futures
from functools import partial

import cv2
import pandas as pd
from rich.table import Table

from src import constants
from src.defaults import CONFIG_DEFAULTS
from src.evaluation import EvaluationConfig, evaluate_concatenated_response
from src.logger import console, logger
from src.template import Template
from src.utils.file import Paths, setup_dirs_for_paths, setup_outputs_for_template
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils, Stats
from src.utils.parsing import get_concatenated_response, open_config_with_defaults

# Import PDF processing utilities
try:
    from utils.image_processing import process_pdf_batch
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False
    logger.warning("PDF processing utilities not available. PDF files will be skipped.")

# Load processors
STATS = Stats()
# Global processing stats
GLOBAL_STATS = {
    "total_files": 0,
    "successful_files": 0,
    "error_files": 0,
    "pdf_files_processed": 0,
    "pdf_pages_extracted": 0,
    "start_time": 0,
}


def entry_point(input_dir, args):
    GLOBAL_STATS["start_time"] = time()
    if not os.path.exists(input_dir):
        raise Exception(f"Given input directory does not exist: '{input_dir}'")
    curr_dir = input_dir
    result = process_dir(input_dir, curr_dir, args)
    
    # Print final summary
    total_time = time() - GLOBAL_STATS["start_time"]
    logger.info("=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total files processed: {GLOBAL_STATS['total_files']}")
    logger.info(f"Successful files: {GLOBAL_STATS['successful_files']}")
    logger.info(f"Files with errors: {GLOBAL_STATS['error_files']}")
    if GLOBAL_STATS['pdf_files_processed'] > 0:
        logger.info(f"PDF files processed: {GLOBAL_STATS['pdf_files_processed']}")
        logger.info(f"PDF pages extracted: {GLOBAL_STATS['pdf_pages_extracted']}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    if GLOBAL_STATS['total_files'] > 0:
        logger.info(f"Average time per file: {total_time/GLOBAL_STATS['total_files']:.4f} seconds")
    logger.info("=" * 50)
    
    return result


def print_config_summary(
    curr_dir,
    omr_files,
    template,
    tuning_config,
    local_config_path,
    evaluation_config,
    args,
):
    # Simplified config summary to reduce log output
    logger.info("")
    logger.info(f"Processing directory: {curr_dir} with {len(omr_files)} images")
    logger.info(f"Template: {template}")
    
    # Only print detailed config in verbose mode
    if tuning_config.get("verbose_logs", False):
        table = Table(title="Current Configurations", show_header=False, show_lines=False)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_row("Directory Path", f"{curr_dir}")
        table.add_row("Count of Images", f"{len(omr_files)}")
        table.add_row("Set Layout Mode ", "ON" if args["setLayout"] else "OFF")
        table.add_row(
            "Markers Detection",
            "ON" if "CropOnMarkers" in template.pre_processors else "OFF",
        )
        table.add_row("Auto Alignment", f"{tuning_config.alignment_params.auto_align}")
        table.add_row("Detected Template Path", f"{template}")
        if local_config_path:
            table.add_row("Detected Local Config", f"{local_config_path}")
        if evaluation_config:
            table.add_row("Detected Evaluation Config", f"{evaluation_config}")

        table.add_row(
            "Detected pre-processors",
            f"{[pp.__class__.__name__ for pp in template.pre_processors]}",
        )
        console.print(table, justify="center")


def process_dir(
    root_dir,
    curr_dir,
    args,
    template=None,
    tuning_config=CONFIG_DEFAULTS,
    evaluation_config=None,
):
    dir_start_time = time()
    
    # Update local tuning_config (in current recursion stack)
    local_config_path = curr_dir.joinpath(constants.CONFIG_FILENAME)
    if os.path.exists(local_config_path):
        tuning_config = open_config_with_defaults(local_config_path)

    # Update local template (in current recursion stack)
    local_template_path = curr_dir.joinpath(constants.TEMPLATE_FILENAME)
    local_template_exists = os.path.exists(local_template_path)
    if local_template_exists:
        template = Template(
            local_template_path,
            tuning_config,
        )
    
    # Look for subdirectories for processing
    subdirs = [d for d in curr_dir.iterdir() if d.is_dir()]

    output_dir = Path(args["output_dir"], curr_dir.relative_to(root_dir))
    paths = Paths(output_dir)

    # Process PDFs if available
    if PDF_PROCESSING_AVAILABLE:
        pdf_files = sorted(curr_dir.glob("*.[pP][dD][fF]"))
        if pdf_files:
            # Process PDFs and add generated images to file list
            logger.info(f"Found {len(pdf_files)} PDF files in {curr_dir}")
            extracted_images = process_directory_pdfs(pdf_files, curr_dir, tuning_config)
            logger.info(f"Extracted {len(extracted_images)} images from PDFs")
            # The extracted images will be found in the next image search step
            
    # look for images in current dir to process
    exts = ("*.[pP][nN][gG]", "*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]")
    omr_files = sorted([f for ext in exts for f in curr_dir.glob(ext)])

    # Exclude images (take union over all pre_processors)
    excluded_files = []
    if template:
        for pp in template.pre_processors:
            excluded_files.extend(Path(p) for p in pp.exclude_files())

    local_evaluation_path = curr_dir.joinpath(constants.EVALUATION_FILENAME)
    if not args["setLayout"] and os.path.exists(local_evaluation_path):
        if not local_template_exists:
            logger.warning(
                f"Found an evaluation file without a parent template file: {local_evaluation_path}"
            )
        evaluation_config = EvaluationConfig(
            curr_dir,
            local_evaluation_path,
            template,
            tuning_config,
        )

        excluded_files.extend(
            Path(exclude_file) for exclude_file in evaluation_config.get_exclude_files()
        )

    omr_files = [f for f in omr_files if f not in excluded_files]

    if omr_files:
        if not template:
            logger.error(
                f"Found images, but no template in the directory tree \
                of '{curr_dir}'. \nPlace {constants.TEMPLATE_FILENAME} in the \
                appropriate directory."
            )
            raise Exception(
                f"No template file found in the directory tree of {curr_dir}"
            )

        setup_dirs_for_paths(paths)
        outputs_namespace = setup_outputs_for_template(paths, template)

        print_config_summary(
            curr_dir,
            omr_files,
            template,
            tuning_config,
            local_config_path,
            evaluation_config,
            args,
        )
        if args["setLayout"]:
            show_template_layouts(omr_files, template, tuning_config)
        else:
            process_files(
                omr_files,
                template,
                tuning_config,
                evaluation_config,
                outputs_namespace,
            )

    elif not subdirs:
        # Each subdirectory should have images or should be non-leaf
        logger.info(
            f"No valid images or sub-folders found in {curr_dir}.\
            Empty directories not allowed."
        )

    # recursively process sub-folders
    for d in subdirs:
        process_dir(
            root_dir,
            d,
            args,
            template,
            tuning_config,
            evaluation_config,
        )
    
    dir_total_time = time() - dir_start_time
    logger.info(f"Directory processing completed in {dir_total_time:.2f} seconds: {curr_dir}")
    return "Success"


def process_directory_pdfs(pdf_files, output_dir, tuning_config):
    """
    Process all PDF files in a directory and convert them to images
    
    Args:
        pdf_files: List of PDF file paths
        output_dir: Directory to save extracted images
        tuning_config: Configuration settings
        
    Returns:
        List of paths to extracted images
    """
    # Skip if no PDF processing available
    if not PDF_PROCESSING_AVAILABLE or not pdf_files:
        return []
    
    # Get batch size from file count
    batch_size = len(pdf_files)
    
    # Get performance settings based on batch size
    try:
        from utils.batch_config import get_batch_profile
        perf_settings = get_batch_profile(batch_size)
        dpi = perf_settings.get("dpi", 100)
        quality = perf_settings.get("quality", 70)
    except ImportError:
        # Default settings if batch_config is not available
        if batch_size <= 5:
            dpi, quality = 120, 80
        elif batch_size <= 20:
            dpi, quality = 100, 70
        elif batch_size <= 50:
            dpi, quality = 90, 65
        else:
            dpi, quality = 80, 60
    
    logger.info(f"Processing {len(pdf_files)} PDF files with DPI={dpi}, quality={quality}")
    
    pdf_start_time = time()
    
    # Create subdirectory for converted images
    pdf_images_dir = Path(output_dir) / "pdf_extracted_images"
    pdf_images_dir.mkdir(exist_ok=True)
    
    # Process PDFs in batch mode
    results = process_pdf_batch(pdf_files, pdf_images_dir, dpi, quality)
    
    # Collect all extracted image paths
    all_images = []
    for pdf_path, image_paths in results.items():
        pdf_file_name = Path(pdf_path).name
        GLOBAL_STATS["pdf_files_processed"] += 1
        GLOBAL_STATS["pdf_pages_extracted"] += len(image_paths)
        all_images.extend(image_paths)
        logger.info(f"PDF {pdf_file_name}: extracted {len(image_paths)} pages")
    
    pdf_total_time = time() - pdf_start_time
    pages_per_second = len(all_images) / pdf_total_time if pdf_total_time > 0 else 0
    
    logger.info(f"PDF processing completed in {pdf_total_time:.2f} seconds")
    logger.info(f"Extracted {len(all_images)} pages from {len(pdf_files)} PDFs ({pages_per_second:.2f} pages/sec)")
    
    return all_images


def show_template_layouts(omr_files, template, tuning_config):
    for file_path in omr_files:
        file_name = file_path.name
        file_path = str(file_path)
        in_omr = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        in_omr = template.image_instance_ops.apply_preprocessors(
            file_path, in_omr, template
        )
        template_layout = template.image_instance_ops.draw_template_layout(
            in_omr, template, shifted=False, border=2
        )
        InteractionUtils.show(
            f"Template Layout: {file_name}", template_layout, 1, 1, config=tuning_config
        )


def process_files(
    omr_files,
    template,
    tuning_config,
    evaluation_config,
    outputs_namespace,
):
    global GLOBAL_STATS
    start_time = time()
    files_counter = len(omr_files)
    STATS.files_not_moved = 0
    GLOBAL_STATS["total_files"] += files_counter
    
    # Determine optimal batch size and worker count
    try:
        # Try to import batch settings if available
        from utils.batch_config import BATCH_SIZES, RESOURCE_LIMITS
        batch_size = BATCH_SIZES.get("omr_file_chunk", 20)
        max_workers = RESOURCE_LIMITS.get("max_omr_workers", min(os.cpu_count(), 8))
    except ImportError:
        # Default settings if batch_config is not available
        batch_size = min(20, len(omr_files))
        max_workers = min(os.cpu_count() or 4, 8)  # Use at most 8 workers
    
    # Use parallel processing if more than 5 files and parallel processing is enabled
    use_parallel = files_counter > 5 and tuning_config.get("parallel_processing", True)
    
    if use_parallel:
        # Process files in parallel using process_single_file function
        logger.info(f"Processing {files_counter} files in parallel with {max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create partial function with fixed arguments
            process_func = partial(
                process_single_file,
                template=template,
                tuning_config=tuning_config,
                evaluation_config=evaluation_config,
                outputs_namespace=outputs_namespace
            )
            
            # Submit all tasks to the executor
            future_to_file = {executor.submit(process_func, file_path): file_path for file_path in omr_files}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                success = future.result()
                if success:
                    GLOBAL_STATS["successful_files"] += 1
                else:
                    GLOBAL_STATS["error_files"] += 1
    else:
        # Process files sequentially for small batches
        for file_path in omr_files:
            success = process_single_file(
                file_path, 
                template, 
                tuning_config, 
                evaluation_config, 
                outputs_namespace
            )
            if success:
                GLOBAL_STATS["successful_files"] += 1
            else:
                GLOBAL_STATS["error_files"] += 1

    total_time = time() - start_time
    logger.info(f"Processing completed in {total_time:.2f} seconds for {files_counter} files")


def process_single_file(
    file_path,
    template,
    tuning_config,
    evaluation_config,
    outputs_namespace,
):
    """Process a single OMR file and return True if successful, False if error"""
    try:
        file_name = file_path.name

        in_omr = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

        template.image_instance_ops.reset_all_save_img()
        template.image_instance_ops.append_save_img(1, in_omr)

        # Preprocessing step
        in_omr = template.image_instance_ops.apply_preprocessors(
            file_path, in_omr, template
        )

        if in_omr is None:
            # Error OMR case
            new_file_path = outputs_namespace.paths.errors_dir.joinpath(file_name)
            outputs_namespace.OUTPUT_SET.append(
                [file_name] + outputs_namespace.empty_resp
            )
            if check_and_move(
                constants.ERROR_CODES.NO_MARKER_ERR, file_path, new_file_path
            ):
                err_line = [
                    file_name,
                    file_path,
                    new_file_path,
                    "NA",
                ] + outputs_namespace.empty_resp
                pd.DataFrame(err_line, dtype=str).T.to_csv(
                    outputs_namespace.files_obj["Errors"],
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            return False
            
        # Score calculation
        shifted_result = template.image_instance_ops.apply_template(
            in_omr, template, file_name
        )

        concatenated_response = get_concatenated_response(shifted_result)
        result_copy = [file_name]

        if tuning_config.show_image_level:
            final_marked = template.image_instance_ops.get_concatenated_response_img(shifted_result)
            template.image_instance_ops.append_save_img(6, final_marked)

        if tuning_config.save_marked_images:
            if template.image_instance_ops.save_img_list[-1] is not None:
                output_path = outputs_namespace.paths.storage_dir.joinpath(file_name)
                cv2.imwrite(
                    str(output_path), template.image_instance_ops.save_img_list[-1]
                )
        
        save_response = template.image_instance_ops.save_responses(
            shifted_result,
            file_name,
            outputs_namespace.files_obj,
            outputs_namespace.OUTPUT_SET,
            result_copy,
            concatenated_response,
        )
        
        # Evaluation
        if evaluation_config:
            file_type, score, resp = evaluate_concatenated_response(
                concatenated_response, evaluation_config
            )
            
            # Based on evaluation
            if file_type == constants.ERROR_CODES.MULTI_BUBBLE_WARN:
                check_and_move(
                    file_type,
                    file_path,
                    outputs_namespace.paths.multi_marked_dir.joinpath(file_name),
                )
            elif file_type == constants.ERROR_CODES.NO_MARKER_ERR:
                check_and_move(
                    file_type,
                    file_path,
                    outputs_namespace.paths.errors_dir.joinpath(file_name),
                )
            elif file_type == constants.ERROR_CODES.INVALID_PAGE:
                check_and_move(
                    file_type,
                    file_path,
                    outputs_namespace.paths.invalid_dir.joinpath(file_name),
                )
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {str(e)}")
        return False


def check_and_move(error_code, file_path, filepath2):
    # TODO: fix file movement into error/multimarked/invalid etc again
    return True 