import cv2
import io
import os
import time
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def transform_image(image, IMAGE_SIZE):
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ]
    )
    return transform_pipeline(image).unsqueeze(0)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def overlay_detected_mask(image):
    mask_image = image.resize((256,256)).convert("RGBA")
    output_im = output[-1][0][0, :, :]
    for k in range(len(output[-1])):
        output_im2 = output[-1][k][0, :, :]
        output_im2[output_im2 > 0.5] = 1
        output_im2[output_im2 < 0.5] = 0
        output_im = output_im + output_im2
    output_im[output_im > 0.5] = 255
    output_im[output_im < 0.5] = 0

    masks = Image.fromarray(output_im).convert("RGBA")

    width = masks.size[0]
    height = masks.size[1]
    for i in range(0, width):  # process all pixels
        for j in range(0, height):
            data = masks.getpixel((i, j))
            if data[:3] == (255, 255, 255):
                masks.putpixel((i, j), (210, 4, 45))
            else:
                masks.putpixel((i, j), (0, 0, 0, 0))

    # overlay = Image.blend(mask_image, masks, alpha=0.2)
    overlay = Image.alpha_composite(mask_image, masks)
    masks.close()
    mask_image.close()

    width, height = image.size
    overlay = overlay.resize((width, height))  # .convert("RGB")
    overlay.show()


def perform_secondary_cv_detection(image, output):
    orig_image = image.copy()
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    (thresh, blackAndWhiteImage) = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(blackAndWhiteImage, 200, 650)  # 150, 650

    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply the dilation operation to the edged image
    dilate = cv2.dilate(edged, kernel, iterations=1)
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

    try:
        print('Using Hough Circle Edge Detection')
        circles = cv2.HoughCircles(closing, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=int((256 / 2) - 50), maxRadius=int(256 / 2))

        circles = np.uint8(np.around(circles))
        mask = np.zeros(open_cv_image.shape[:2], dtype='uint8')
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)

        masked = cv2.bitwise_and(open_cv_image, open_cv_image, mask=mask)
        detected_circle = Image.fromarray(masked).convert('RGB')

        return detected_circle
    except TypeError:
        print("Wasn't Able to Find Any Additional Contours")
        return orig_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path',dest='dataset_path',
                        help='Parent folder with folders to process (usually source data in trainA (images) & trainB (masks))')
    parser.add_argument('--resize', dest='target_image_size',
                        help='Sets image size to align with MRCNN Model Input')
    parser.add_argument('--preprocess_model', dest='preprocess_model_path',
                        help='Path and filename for the ONNX model used to preprocess source images (if needed)')
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    image_size = int(args.target_image_size)
    onnx_model = args.preprocess_model_path

    image_directory = sorted(os.listdir(dataset_path.joinpath('trainA')))
    mask_directory = sorted(os.listdir(dataset_path.joinpath('trainB')))

    for image, mask in zip(image_directory, mask_directory):

        image_name = dataset_path.joinpath('trainA').joinpath(str(image))
        mask_name = dataset_path.joinpath('trainB').joinpath(str(mask))

        with open(image_name, 'rb') as reader:
            file_bytes = reader.read()

        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        mask = Image.open(mask_name).convert("RGBA")

        orig_image_size = img.size

        img_tensor = transform_image(img, image_size)
        mask = mask.resize((image_size, image_size))

        ort_session = onnxruntime.InferenceSession('mrcnn_circle.onnx')

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_tensor)}

        ort_outs = ort_session.run(None, ort_inputs)

        # USE BOUNDING BOX TO CROP NEW IMAGE
        # PASS INTO SECONDARY CIRCLE DETECTION
        sample_found = img.resize((image_size, image_size)).copy()
        box_found = ort_outs[0][0]
        sample_found = sample_found.crop(box_found)
        mask_found = mask.crop(box_found)

        final_image = perform_secondary_cv_detection(sample_found, ort_outs)
        sample_found.close()

        final_image = final_image.resize((image_size, image_size))
        mask_found = mask_found.resize((image_size, image_size))

        final_image.save(image_name)
        mask_found.save(mask_name)

        final_image.close()
        mask_found.close()
