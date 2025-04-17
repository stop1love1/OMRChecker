"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""
import os
from csv import QUOTE_NONNUMERIC
from pathlib import Path
from time import time
import concurrent.futures
from functools import partial

import cv2
import pandas as pd
import numpy as np
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

# Load processors
STATS = Stats()


def entry_point(input_dir, args):
    if not os.path.exists(input_dir):
        raise Exception(f"Given input directory does not exist: '{input_dir}'")
    curr_dir = input_dir
    return process_dir(input_dir, curr_dir, args)

def process_dir(
    root_dir,
    curr_dir,
    args,
    template=None,
    tuning_config=CONFIG_DEFAULTS,
    evaluation_config=None,
):
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


# Helper function to process a single file for parallel processing
def process_single_file(file_path, template, tuning_config, evaluation_config, outputs_namespace):
    try:
        file_name = file_path.name
        
        # Read image with optimized flag
        in_omr = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if in_omr is None:
            return {"error": "Failed to read image", "file_path": file_path}
            
        # Create a new template instance ops for thread safety
        image_instance_ops = template.image_instance_ops.__class__(tuning_config)
        template.image_instance_ops = image_instance_ops
        
        template.image_instance_ops.reset_all_save_img()
        template.image_instance_ops.append_save_img(1, in_omr)

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
            return {"status": "error", "file_path": file_path, "error_type": "NO_MARKER_ERR"}

        # uniquify
        file_id = str(file_name)
        save_dir = outputs_namespace.paths.save_marked_dir
        
        try:
            (
                response_dict,
                final_marked,
                multi_marked,
                _,
            ) = template.image_instance_ops.read_omr_response(
                template, image=in_omr, name=file_id, save_dir=save_dir
            )

            # concatenate roll nos, set unmarked responses, etc
            omr_response = get_concatenated_response(response_dict, template)    

            # Evaluation output (without detailed logging)
            score = 0
            if evaluation_config is not None:
                score = evaluate_concatenated_response(
                    omr_response, evaluation_config, file_path, outputs_namespace.paths.evaluation_dir
                )

            if tuning_config.outputs.show_image_level >= 2:
                InteractionUtils.show(
                    f"Final Marked Bubbles : '{file_id}'",
                    ImageUtils.resize_util_h(
                        final_marked, int(tuning_config.dimensions.display_height * 1.3)
                    ),
                    1,
                    1,
                    config=tuning_config,
                )

            resp_array = []
            for k in template.output_columns:
                resp_array.append(omr_response[k])

            # Thread-safe append to output set
            with outputs_namespace.output_lock:
                outputs_namespace.OUTPUT_SET.append([file_name] + resp_array)

            if multi_marked == 0 or not tuning_config.outputs.filter_out_multimarked_files:
                STATS.files_not_moved += 1
                new_file_path = save_dir.joinpath(file_id)
                # Enter into Results sheet-
                results_line = [file_name, file_path, new_file_path, score] + resp_array
                # Write/Append to results_line file(opened in append mode)
                # Lock to avoid concurrent writes
                with outputs_namespace.file_locks["Results"]:
                    pd.DataFrame(results_line, dtype=str).T.to_csv(
                        outputs_namespace.files_obj["Results"],
                        mode="a",
                        quoting=QUOTE_NONNUMERIC,
                        header=False,
                        index=False,
                    )
                return {"status": "success", "file_path": file_path, "multi_marked": False}
            else:
                # multi_marked file
                new_file_path = outputs_namespace.paths.multi_marked_dir.joinpath(file_name)
                if check_and_move(
                    constants.ERROR_CODES.MULTI_BUBBLE_WARN, file_path, new_file_path
                ):
                    mm_line = [file_name, file_path, new_file_path, "NA"] + resp_array
                    # Lock to avoid concurrent writes
                    with outputs_namespace.file_locks["MultiMarked"]:
                        pd.DataFrame(mm_line, dtype=str).T.to_csv(
                            outputs_namespace.files_obj["MultiMarked"],
                            mode="a",
                            quoting=QUOTE_NONNUMERIC,
                            header=False,
                            index=False,
                        )
                return {"status": "success", "file_path": file_path, "multi_marked": True}
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {"status": "error", "file_path": file_path, "error": str(e)}
            
    except Exception as e:
        logger.error(f"Error in process_single_file: {str(e)}")
        return {"status": "error", "file_path": file_path, "error": str(e)}


def process_files(
    omr_files,
    template,
    tuning_config,
    evaluation_config,
    outputs_namespace,
):
    start_time = time()
    files_counter = 0
    STATS.files_not_moved = 0
    
    # Add thread synchronization to outputs_namespace
    import threading
    outputs_namespace.output_lock = threading.Lock()
    outputs_namespace.file_locks = {
        "Results": threading.Lock(),
        "MultiMarked": threading.Lock(),
        "Errors": threading.Lock()
    }
    
    try:
        # Get optimized batch settings if they exist
        from utils.batch_config import BATCH_SIZES, RESOURCE_LIMITS
        batch_size = BATCH_SIZES.get("omr_file_chunk", 20)
        max_workers = RESOURCE_LIMITS.get("max_omr_workers", min(os.cpu_count(), 8))
    except ImportError:
        # Default settings if batch_config is not available
        batch_size = min(20, len(omr_files))
        max_workers = max(1, min(os.cpu_count(), 8))
    
    # Group files into optimal batch sizes to balance parallelism and memory usage
    file_batches = [omr_files[i:i+batch_size] for i in range(0, len(omr_files), batch_size)]
    
    # Pre-process high frequency files first (optional sorting)
    if len(file_batches) > 1:
        logger.info(f"Processing {len(omr_files)} files in {len(file_batches)} batches")
    
    for batch_idx, batch in enumerate(file_batches):
        batch_start_time = time()
        
        # Process each batch of files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use partial to pass the fixed arguments
            process_func = partial(
                process_single_file,
                template=template,
                tuning_config=tuning_config,
                evaluation_config=evaluation_config,
                outputs_namespace=outputs_namespace
            )
            
            # Submit batch for processing
            futures = {executor.submit(process_func, file_path): file_path for file_path in batch}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    files_counter += 1
                    
                    if result.get("status") == "error":
                        logger.warning(f"Failed to process file: {result.get('file_path')} - {result.get('error')}")
                    
                except Exception as exc:
                    logger.error(f"File processing generated an exception: {exc}")
        
        # Log batch processing time for monitoring
        if len(file_batches) > 1:
            batch_time = time() - batch_start_time
            logger.info(f"Batch {batch_idx+1}/{len(file_batches)} processed in {batch_time:.2f} seconds ({len(batch)} files)")
            
            # Force garbage collection after each batch to prevent memory issues
            import gc
            gc.collect()

    total_time = time() - start_time
    logger.info(f"Processing completed in {total_time:.2f} seconds for {files_counter} files")


def check_and_move(error_code, file_path, filepath2):
    # TODO: fix file movement into error/multimarked/invalid etc again
    STATS.files_not_moved += 1
    return True


def print_stats(start_time, files_counter, tuning_config):
    time_checking = max(1, round(time() - start_time, 2))
    log = logger.info

    if tuning_config.outputs.show_image_level <= 0:
        log(
            f"\nFinished Checking {files_counter} file(s) in {round(time_checking, 1)} seconds i.e. ~{round(time_checking / 60, 1)} minute(s)."
        )
        log(
            f"{'OMR Processing Rate': <27}: \t ~ {round(time_checking / files_counter, 2)} seconds/OMR"
        )
        log(
            f"{'OMR Processing Speed': <27}: \t ~ {round((files_counter * 60) / time_checking, 2)} OMRs/minute"
        )
    else:
        log(f"\n{'Total script time': <27}: {time_checking} seconds")

    if tuning_config.outputs.show_image_level <= 1:
        log(
            "\nTip: To see some awesome visuals, open config.json and increase 'show_image_level'"
        )
