"""
Entry file with detailed timing logs for OMRChecker

Modified version of original entry.py with timing information.
"""
import os
from csv import QUOTE_NONNUMERIC
from pathlib import Path
from time import time

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

# Load processors
STATS = Stats()


def entry_point(input_dir, args):
    if not os.path.exists(input_dir):
        raise Exception(f"Given input directory does not exist: '{input_dir}'")
    curr_dir = input_dir
    return process_dir(input_dir, curr_dir, args)


def print_config_summary(
    curr_dir,
    omr_files,
    template,
    tuning_config,
    local_config_path,
    evaluation_config,
    args,
):
    logger.info("")
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
    logger.info(f"[TIMING] Starting directory processing: {curr_dir}")
    dir_start_time = time()
    
    # Update local tuning_config (in current recursion stack)
    config_start_time = time()
    local_config_path = curr_dir.joinpath(constants.CONFIG_FILENAME)
    if os.path.exists(local_config_path):
        logger.info(f"[TIMING] Loading local config: {local_config_path}")
        tuning_config = open_config_with_defaults(local_config_path)
    config_time = time() - config_start_time
    logger.info(f"[TIMING] Config loading took {config_time:.2f} seconds")

    # Update local template (in current recursion stack)
    template_start_time = time()
    local_template_path = curr_dir.joinpath(constants.TEMPLATE_FILENAME)
    local_template_exists = os.path.exists(local_template_path)
    if local_template_exists:
        logger.info(f"[TIMING] Loading template: {local_template_path}")
        template = Template(
            local_template_path,
            tuning_config,
        )
    template_time = time() - template_start_time
    logger.info(f"[TIMING] Template loading took {template_time:.2f} seconds")
    
    # Look for subdirectories for processing
    subdirs = [d for d in curr_dir.iterdir() if d.is_dir()]

    output_dir = Path(args["output_dir"], curr_dir.relative_to(root_dir))
    paths = Paths(output_dir)

    # look for images in current dir to process
    file_find_start = time()
    exts = ("*.[pP][nN][gG]", "*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]")
    omr_files = sorted([f for ext in exts for f in curr_dir.glob(ext)])
    file_find_time = time() - file_find_start
    logger.info(f"[TIMING] Found {len(omr_files)} image files in {file_find_time:.2f} seconds")

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

        dir_setup_start = time()
        setup_dirs_for_paths(paths)
        outputs_namespace = setup_outputs_for_template(paths, template)
        dir_setup_time = time() - dir_setup_start
        logger.info(f"[TIMING] Directory setup took {dir_setup_time:.2f} seconds")

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
    logger.info(f"[TIMING] Directory processing completed in {dir_total_time:.2f} seconds: {curr_dir}")
    return "Success"


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
    start_time = int(time())
    logger.info(f"[TIMING] Starting to process {len(omr_files)} files")
    files_counter = 0
    STATS.files_not_moved = 0
    
    total_preprocessing_time = 0
    total_scoring_time = 0
    total_reading_time = 0
    total_moving_time = 0

    for file_path in omr_files:
        file_start_time = time()
        files_counter += 1
        file_name = file_path.name

        read_start_time = time()
        in_omr = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        read_time = time() - read_start_time
        total_reading_time += read_time

        logger.info("")
        logger.info(
            f"({files_counter}) Opening image: \t'{file_path}'\tResolution: {in_omr.shape}"
        )

        template.image_instance_ops.reset_all_save_img()
        template.image_instance_ops.append_save_img(1, in_omr)

        # Preprocessing step
        preprocess_start_time = time()
        in_omr = template.image_instance_ops.apply_preprocessors(
            file_path, in_omr, template
        )
        preprocess_time = time() - preprocess_start_time
        total_preprocessing_time += preprocess_time
        logger.info(f"[TIMING] Preprocessing of {file_name} took {preprocess_time:.2f} seconds")

        if in_omr is None:
            # Error OMR case
            moving_start_time = time()
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
            moving_time = time() - moving_start_time
            total_moving_time += moving_time
            logger.info(f"[TIMING] Moving error file took {moving_time:.2f} seconds")
            file_total_time = time() - file_start_time
            logger.info(f"[TIMING] Processing of error file {file_name} took {file_total_time:.2f} seconds")
            continue
            
        # Score calculation
        score_start_time = time()
        shifted_result = template.image_instance_ops.apply_template(
            in_omr, template, file_name
        )
        score_time = time() - score_start_time
        total_scoring_time += score_time
        logger.info(f"[TIMING] Scoring of {file_name} took {score_time:.2f} seconds")

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
            moving_start_time = time()
            file_type, score, resp = evaluate_concatenated_response(
                concatenated_response, evaluation_config
            )
            moving_time = time() - moving_start_time
            total_moving_time += moving_time
            
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

            logger.info(f"[{file_type}]\t{file_name}: {score}")
            if resp:
                logger.info(f"Resp: {resp}")
                
        file_total_time = time() - file_start_time
        logger.info(f"[TIMING] Total processing of {file_name} took {file_total_time:.2f} seconds")

    # Print timing statistics
    logger.info("")
    logger.info(f"[TIMING] Processing Summary for {files_counter} files:")
    logger.info(f"[TIMING] Average reading time: {(total_reading_time/files_counter if files_counter else 0):.2f} sec/file")
    logger.info(f"[TIMING] Average preprocessing time: {(total_preprocessing_time/files_counter if files_counter else 0):.2f} sec/file")
    logger.info(f"[TIMING] Average scoring time: {(total_scoring_time/files_counter if files_counter else 0):.2f} sec/file")
    logger.info(f"[TIMING] Average file movement time: {(total_moving_time/files_counter if files_counter else 0):.2f} sec/file")
    logger.info(f"[TIMING] Total time: {time() - start_time:.2f} seconds")

    print_stats(start_time, files_counter, tuning_config)


def check_and_move(error_code, file_path, filepath2):
    # TODO: fix file movement into error/multimarked/invalid etc again
    return True


def print_stats(start_time, files_counter, tuning_config):
    end_time = int(time())
    time_taken = end_time - start_time
    files_not_moved = STATS.files_not_moved

    logger.info("")
    logger.info(f"{files_counter} files processed in {time_taken}s.")
    if files_counter > 0:
        logger.info(f"({time_taken/files_counter:.2f}s per file)")
    if files_not_moved > 0:
        logger.info(f"Warning: {files_not_moved} files not moved.")
    if tuning_config.show_image_level:
        logger.warning(f"Note: showing images may slow down the program.") 