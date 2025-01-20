#!/usr/bin/env python3
"""
    GENERALIZED SCRIPT TO CREATE A CV MODEL FOR AN APPLICATION
        INPUTS:
            - Folder of Source Images with Labels
                - Used in GAN Training
                - Labels consist of Single Color Mask
            - Folder of Backgrounds & Foregrounds
                - Used in Brute Force Mask Generation (COCO method)
        OUTPUTS:
            - GAN Model for generating synthetic samples
            - MRCNN Model for detecting/segmenting objects of interest

    WORKFLOW:
        1. Use Labeled Images to train a GAN Generative model
        2. Use brute force synthetics to generate masks to provide input to trained GAN
        3. Use Generated Synthetics to train a MRCNN Model
            - need to adjust output from GAN. Adjust from Single Color to Multi-Color
"""
import os
import pathlib
import shutil
import time
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import subprocess


def convert_arguments(argument_list):
    return [str(arg) for arg in argument_list]


BASE_DIR = pathlib.Path(__file__).parent.parent
CUR_DIR = pathlib.Path(__file__).parent

## SETUP MODEL & DATA DIRECTORY OUTSIDE REPO FOLDER IN SAME PARENT
MODEL_DIR = BASE_DIR.joinpath('CV_MODELS')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR.joinpath('CV_DATA')
DATA_DIR.mkdir(parents=True, exist_ok=True)

Application_Name = 'chip'           # provide a unique application name to generate all folders for workflow
IMAGE_SIZE = 256                    # set this to whatever image size both the gan and mrcnn will use
DATASET_SIZE = 50

PREPROCESS_SOURCE_IMAGES = False
BUILD_GAN_IMAGES = False            # If GAN aligned images haven't been generated use this flag
TRAIN_GAN = True
COCO_GENERATE = True
GAN_GENERATE = True
PREPARE_SYNTHETIC_DATA = True
TRAIN_MRCNN = True
SHOW_MRCNN = True
CONVERT_ONNX = True

# SETUP GAN SAVE PATH
GAN_CHECKPOINTS = MODEL_DIR.joinpath(f'{Application_Name}').joinpath('pytorch').joinpath('gan')
MRCNN_CHECKPOINTS = MODEL_DIR.joinpath(f'{Application_Name}').joinpath('pytorch').joinpath('mrcnn')
ONNX_CHECKPOINTS = MODEL_DIR.joinpath(f'{Application_Name}').joinpath('onnx')
### IMPORT DATASETS --> BOTH LABELED SOURCE DATA & COCO FORMAT
### *_source_data --> Houses the trainA/testA & trainB/testB folders representing images and masks respectively and combines into train/test
### *_coco_data --> Houses the brute force synthetic masks generation for input to the gan using ImageComposition code
### *_gan_data --> Houses the output files from the GAN model, masks converted to synthetic examples
labeled_source_data = DATA_DIR.joinpath(f'{Application_Name}_source_data')
coco_source_data = DATA_DIR.joinpath(f'{Application_Name}_coco_data')
gan_synthetic_data = DATA_DIR.joinpath(f'{Application_Name}_gan_data')

### Generate folders as needed
labeled_source_data.mkdir(parents=True, exist_ok=True)
coco_source_data.mkdir(parents=True, exist_ok=True)
gan_synthetic_data.mkdir(parents=True, exist_ok=True)
GAN_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
MRCNN_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
ONNX_CHECKPOINTS.mkdir(parents=True, exist_ok=True)


if PREPROCESS_SOURCE_IMAGES:
    dataset_preprocess_arguments = [
        '--dataset-path', labeled_source_data,
        '--resize', IMAGE_SIZE,
        '--preprocess_model', 'mrcnn_circle.onnx'
    ]

    try:
        dataset_preprocess_arguments = convert_arguments(dataset_preprocess_arguments)
        dataset_process = subprocess.run(["python3.9", "preprocess_source_images_for_gan.py"]+dataset_preprocess_arguments,
                                        check=True, capture_output=False, text=True)
        print(dataset_process)
    except subprocess.CalledProcessError as e:
        print(e)


# USE PIX2PIX structure to convert images and masks into aligned image|mask single images for GAN TRAINING

## USE RICHIES CODE TO
if BUILD_GAN_IMAGES:
    #### RICHIE: Script to take Supervisely --> gan_source_data (orig size) --> Images, Machine Masks (converted to b/w)
    ####         Build GAN Aligned dataset (Resize to MRCNN IMAGE TRAINING SIZE) --> combine_A_and_B.py --> gan_combined_data
    ####         Extracted backgrounds/foregrounds into coco_source_data
    ####    may need to add build folders and such (generalize to application / gan_source & coco_source / with sub folders
    dataset_alignment_arguments = [
        '--dataset-path', labeled_source_data,
        '--resize', IMAGE_SIZE
    ]

    try:
        dataset_alignment_arguments = convert_arguments(dataset_alignment_arguments)
        gan_align_data = subprocess.run(["python3.9", "make_dataset_aligned.py"]+dataset_alignment_arguments,
                                        check=True, capture_output=False, text=True)
        print(gan_align_data)
    except subprocess.CalledProcessError as e:
        print(e)


#### UPDATE TO SAVE LESS FREQUENTLY --> SCALED TO NUMBER OF EPOCHS
# USE PIX2PIX structure to use aligned image|mask dataset to TRAIN GAN
# TRAINED GAN provides a way to convert inputs into Generated Synthetic Images mimicking the dataset in question
# GOAL Is to use BRUTE FORCE MASKS as INPUTS to the TRAINED GAN and generate REALISTIC SYNTHETIC VERSIONS
if TRAIN_GAN:
    ### Initialize GAN model and Train on Labeled Source Data
    gan_training_arguments = [
        '--dataroot', labeled_source_data,
        '--checkpoints_dir', GAN_CHECKPOINTS,
        '--name', Application_Name+'_pix2pix',
        '--model', 'pix2pix',
        '--direction', 'BtoA',
        '--n_epochs', 1,
        '--n_epochs_decay', 1,
        '--gpu_ids', '-1',
        '--preprocess', False,
    ]

    try:
        gan_training_arguments = convert_arguments(gan_training_arguments)
        gan_training = subprocess.run(["python3.9", "train_gan.py"]+gan_training_arguments,
                                  check=True, capture_output=False, text=True)
        print(gan_training)
    except subprocess.CalledProcessError as e:
        print(e)


# RUN COCO BRUTE FORCE IMAGE COMPOSITION TO ARTIFICIALLY CONTROL DATASET DISTRIBUTIONS
# MASKS OUTPUT by this brute force approach provide the generated masks as inputs for a TRAINED GAN
# COCO is a DATASET STRUCTURE DEVELOPED BY MICROSOFT FOR OBJECT DETECTION MODELS
# SYSTEM GENERATES TRAINING - VALIDATION SET WITH AN .8 - .2 SPLITS, system will always generate at least a .1 val set
if COCO_GENERATE:
    ### Generate COCO dataset of  brute force generated masks --> INPUT TO Trained GAN For synthetic dataset
    coco_generation_arguments = [
        '--source_data', coco_source_data,
        '--dataset_size', DATASET_SIZE,
        '--image_size', IMAGE_SIZE,
        '--split_ratio_train', .8
    ]
    coco_generation_arguments = convert_arguments(coco_generation_arguments)
    rough_gen = subprocess.run(["python3.9", "create_synthetic_coco_masks.py"]+coco_generation_arguments,
                               check=True, capture_output=False, text=True)
    print(rough_gen)


# USE TRAINED GAN AND COCO BRUTE FORCE IMAGE MASKS TO ARTIFICALLY GENERATE ANY SIZE DATASET DESIRED
# MRCNN EXPECTS TRAINING AND VALIDATION SET
### Feed in COCO masks to GAN to generate 'realistic' synthetic training data
### RUN GAN on both TRAIN AND VAL Folders
coco_masks_path_train = coco_source_data.joinpath('coco_brute_force').joinpath('train').joinpath('masks')
coco_masks_path_val = coco_source_data.joinpath('coco_brute_force').joinpath('val').joinpath('masks')
gan_result_train = gan_synthetic_data.joinpath('coco_results_train')
gan_result_val = gan_synthetic_data.joinpath('coco_results_val')

coco_set = [coco_masks_path_train, coco_masks_path_val]
gan_set = [gan_result_train, gan_result_val]

if GAN_GENERATE:
    start = time.time()
    coco_gan_generation_arguments = [
        '--dataroot', coco_set[0],
        '--checkpoints_dir', GAN_CHECKPOINTS,
        '--num_test', len(os.listdir(coco_set[0])),
        '--name', Application_Name+'_pix2pix',
        '--model', 'test',
        '--netG', 'unet_256',
        '--direction', 'AtoB',
        '--norm', 'batch',
        '--gpu_ids', '-1',
        '--results_dir', gan_set[0]
    ]

    coco_gan_generation_arguments = convert_arguments(coco_gan_generation_arguments)
    gan_train_generator = subprocess.run(["python3.9", "test_gan.py"]+coco_gan_generation_arguments,
                                         check=True, capture_output=False, text=True)
    print(gan_train_generator)
    print(f"Generation of {coco_set[0]} ran for: {time.time()-start}")

    start = time.time()
    coco_gan_generation_arguments = [
        '--dataroot', coco_set[1],
        '--num_test', len(os.listdir(coco_set[1])),
        '--checkpoints_dir', GAN_CHECKPOINTS,
        '--name', Application_Name+'_pix2pix',
        '--model', 'test',
        '--netG', 'unet_256',
        '--direction', 'AtoB',
        '--norm', 'batch',
        '--gpu_ids', '-1',
        '--results_dir', gan_set[1]
    ]

    coco_gan_generation_arguments = convert_arguments(coco_gan_generation_arguments)
    gan_val_generator = subprocess.run(["python3.9", "test_gan.py"]+coco_gan_generation_arguments,
                                       check=True, capture_output=False, text=True)
    print(gan_val_generator)
    print(f"Generation of {coco_set[1]} ran for: {time.time()-start}")


if PREPARE_SYNTHETIC_DATA:
    for split in gan_set:
        gan_synthetic_data_input_location = split.joinpath(f'{Application_Name}_pix2pix').joinpath('test_latest').joinpath('images')

        name = 'train' if 'train' in str(split) else 'val'
        split_size = len(os.listdir(gan_synthetic_data_input_location))/2
        gan_set_output_location = gan_synthetic_data.joinpath(f'{name}')
        print(f'Preparing dataset: {name}_{split_size} at {gan_set_output_location}')
        gan_synthetic_data_image_location = gan_set_output_location.joinpath('images')
        gan_synthetic_data_mask_location = gan_set_output_location.joinpath('masks')

        try:
            gan_synthetic_data_image_location.mkdir(parents=True, exist_ok=False)
            gan_synthetic_data_mask_location.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            shutil.rmtree(gan_synthetic_data_image_location)
            shutil.rmtree(gan_synthetic_data_mask_location)
            gan_synthetic_data_image_location.mkdir(parents=True, exist_ok=False)
            gan_synthetic_data_mask_location.mkdir(parents=True, exist_ok=False)

        ### RELOCATE results/application_pix2pix/test_latest/images/
        # into a chip_gan_data folder  -- for both train & validation
        # real (masks) --> train/masks
        # fake (images) --> train/images
        files = sorted(os.listdir(gan_synthetic_data_input_location))
        images = [str(img) for img in files if 'fake' in str(img)]
        masks = [str(mask) for mask in files if 'real' in str(mask)]

        for image, mask in tqdm(zip(images, masks)):
            ### INSERT COLOR CONVERSION HERE
            # Move image to images folder
            shutil.move(gan_synthetic_data_input_location.joinpath(image), gan_synthetic_data_image_location.joinpath(image))
            # Move mask to masks folder
            shutil.move(gan_synthetic_data_input_location.joinpath(mask), gan_synthetic_data_mask_location.joinpath(mask))


if TRAIN_MRCNN:
    train_dir = gan_synthetic_data
    ### Initialize PyTorch MRCNN model for training and Train model
    # # ADD IN SAVING ADJUSTMENT FOR INTERMEDIATE SAVES
    maskrcnn_training_arguments = [
        '--train_dir', gan_synthetic_data.joinpath('train'),
        '--mrcnn_path', MRCNN_CHECKPOINTS,
        '--val_dir', gan_synthetic_data.joinpath('val'),
        '--n_epochs', 2,
        '--name', Application_Name
    ]

    maskrcnn_training_arguments = convert_arguments(maskrcnn_training_arguments)
    maskrcnn_training = subprocess.run(["python3.9", "maskrcnn_training.py"]+maskrcnn_training_arguments,
                                         check=True, capture_output=False, text=True)
    print(maskrcnn_training)


### Any tests or validations we want to perform before finalizing
if SHOW_MRCNN:
    predict_sample_arguments = [
        '--val_dir', gan_synthetic_data.joinpath('val'),
        '--mrcnn_path', MRCNN_CHECKPOINTS,
        '--name', Application_Name
    ]
    predict_sample_arguments = convert_arguments(predict_sample_arguments)
    prediction = subprocess.run(["python3.9", "predict_one_labeled.py"]+predict_sample_arguments,
                                check=True, capture_output=False, text=True).stdout
    print(prediction)


### If Happy, Convert the PyTorch Model to Onnx?
if CONVERT_ONNX:
    onnx_conversion_arguments = [
        '--val_dir', gan_synthetic_data.joinpath('val').joinpath('images'),
        '--mrcnn_path', MRCNN_CHECKPOINTS,
        '--name', Application_Name,
        '--onnx_path', ONNX_CHECKPOINTS,
    ]

    onnx_conversion_arguments = convert_arguments(onnx_conversion_arguments)
    onnx_conversion = subprocess.run(["python3.9", "pytorch_2_onnx.py"]+onnx_conversion_arguments,
                                     check=True, capture_output=False, text=True).stdout
    print(onnx_conversion)
