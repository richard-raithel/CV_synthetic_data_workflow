#!/usr/bin/env python3
import random
import torchvision
import numpy as np
from numpy.testing import assert_allclose
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import argparse
import os
import torch
from pathlib import Path
import onnx
import onnxruntime
from maskrcnn_training import latest_model



BASE_PATH = Path(__file__).parent.parent


def get_model(model_save_path):
    # CHECK DEVICE COMPATIBILITY
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # INITIALIZE MODEL STRUCTURE - HAS TO BE SAME AS MODEL LOADED
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the masks classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the masks predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)

    try:
        print("Searching for previously trained available models...")
        model_nr, folder = latest_model(application=model_save_path)
        save_path = folder.joinpath('model'+str(model_nr)+'.pt')
        model.load_state_dict(torch.load(save_path))
        print("model", str(model_nr), "loaded successfully!")
    except FileNotFoundError:
        print("No previously trained models available")
        pass

    # LOAD TRAINED WEIGHTS INTO INITALIZED MODEL
    model.eval()

    return device, model


def transform_image(image_bytes):
    transform_pipeline = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    image = Image.open(image_bytes).convert("RGB")
    return transform_pipeline(image).unsqueeze(0)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', dest='val_dir', action='store', help='Path to the Validation Data')
    parser.add_argument('--mrcnn_path', dest='checkpoints_dir', action='store', help='Path to the MRCNN Models')
    parser.add_argument('--name', dest='name', action='store', help='Name of CV Application')
    parser.add_argument('--onnx_path', dest='onnx_path', action='store', help='Location to store the ONNX Model')

    arguments = parser.parse_args()

    root_val = arguments.val_dir
    Application_Name = arguments.name
    mrcnn_path = arguments.checkpoints_dir
    onnx_path = arguments.onnx_path

    # Get Model
    device, model = get_model(mrcnn_path)

    # Perform transformations
    random_file = random.choice(os.listdir(root_val))
    sample_file = Path(root_val).joinpath(random_file)
    x = open(sample_file, 'rb')
    x = transform_image(x)

    with torch.no_grad():
        torch_output = model(x)

    model_name = f'mrcnn_{Application_Name}_gan.onnx'

    model_save_path = str(Path(onnx_path).joinpath(model_name))

    torch.onnx.export(model, x, model_save_path,
                      export_params=True, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                      )

    onnx_model = onnx.load(model_save_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(model_save_path)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    cleaned_torch_output = []
    for result in torch_output:
        for key, val in result.items():
            cleaned_torch_output.append(to_numpy(val))
    # print(cleaned_torch_output)

    test_output = assert_allclose(cleaned_torch_output[0], ort_outs[0], rtol=13-3, atol=1e-5, verbose=True)

    print("Exported model has been tested with ONNXRuntime, and is running as expected")
