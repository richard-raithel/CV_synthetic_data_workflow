#!/usr/bin/env python3
import argparse
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import ToPILImage
import warnings
from PIL import Image
from pathlib import Path
from maskrcnn_training import ChipCorrosionDataset, latest_model



def view_mask(targets, images, output, n=1, cmap='Greys'):
    image = images[0]
    src_image = ToPILImage()(image).convert("RGBA")
    src_copy = src_image.copy()

    for i in range(n):
        # plot target (true) masks
        target_im = targets[i]['masks'][0].cpu().numpy()
        for k in range(len(targets[i]['masks'])):
            target_im2 = targets[i]['masks'][k].cpu().numpy()
            target_im2[target_im2 > 0.5] = 1
            target_im2[target_im2 < 0.5] = 0
            target_im = target_im+target_im2

        target_im[target_im > 0.5] = 255
        target_im[target_im < 0.5] = 0

        target_masks = Image.fromarray(target_im).convert("RGBA")

        width = target_masks.size[0]
        height = target_masks.size[1]
        for w in range(0, width):  # process all pixels
            for j in range(0, height):
                data = target_masks.getpixel((w, j))
                if data[:3] == (255,255,255):
                    target_masks.putpixel((w, j), (150, 150, 45))
                else:
                    target_masks.putpixel((w, j), (0, 0, 0, 0))

        target_im = target_im.astype('int64')
        true_cor_area = target_im.sum()

        expected_overlay = Image.alpha_composite(src_copy, target_masks)
        target_masks.close()

        # ax = figure.add_subplot(2, 1, 1)
        # ax.set_title('TRUE corrosion')
        # ax.imshow(target_im, cmap=cmap)
        # # Plot output (predicted) masks
        output_im = output[i]['masks'][0][0, :, :].cpu().detach().numpy()
        num_masks = len(output[i]['masks'])
        max_mask_size = 0
        for k in range(len(output[i]['masks'])):
            output_im2 = output[i]['masks'][k][0, :, :].cpu().detach().numpy()
            output_im2[output_im2 > 0.5] = 1
            output_im2[output_im2 < 0.5] = 0
            output_im = output_im + output_im2

            try:
                width_min = output_im2.nonzero()[0].min()
                width_max = output_im2.nonzero()[0].max()
                width = width_max - width_min
                height_min = output_im2.nonzero()[1].min()
                height_max = output_im2.nonzero()[1].max()
                height = height_max - height_min
                if width > max_mask_size:
                    max_mask_size = width
                elif height > max_mask_size:
                    max_mask_size = height
            except ValueError:
                pass

        output_im[output_im > 0.5] = 255
        output_im[output_im < 0.5] = 0

        masks = Image.fromarray(output_im).convert("RGBA")

        width = masks.size[0]
        height = masks.size[1]
        for i in range(0, width):  # process all pixels
            for j in range(0, height):
                data = masks.getpixel((i, j))
                if data[:3] == (255,255,255):
                    masks.putpixel((i, j), (240, 5, 45))
                else:
                    masks.putpixel((i, j), (0, 0, 0, 0))

        output_im[output_im > 0.5] = 1

        output_im = output_im.astype('int64')
        pred_cor_area = output_im.sum()

        mask_overlay = Image.alpha_composite(src_image, masks)
        masks.close()

        aligned_image = Image.new("RGB", (expected_overlay.size[0] * 2, expected_overlay.size[1]))
        aligned_image.paste(expected_overlay, (0, 0))
        aligned_image.paste(mask_overlay, (expected_overlay.size[0], 0))
        aligned_image.show()

        expected_overlay.close()
        mask_overlay.close()

        return true_cor_area, pred_cor_area, num_masks, max_mask_size



if __name__ == '__main__':
    # ------------------------------- Run Model ----------------------------------------
    # To run the prediction, set the root_val variable to the desired folder location
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', dest='val_dir', help='Set the path to a Validation directory to randomly sample and show')
    parser.add_argument('--mrcnn_path', dest='checkpoints_dir', help='Path to where the MRCNN models are saved')
    parser.add_argument('--name', dest='name', action='store', help='Name of CV Application')

    arguments = parser.parse_args()

    BASE_PATH = Path(__file__).parent.parent

    # set validation data location
    root_val = arguments.val_dir
    model_application = arguments.name
    model_save_path = arguments.checkpoints_dir

    # ------------------------------- Generic Setup --------------------------------------

    # ignore warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # set device to cpu
    device = torch.device('cpu')

    # ------------------------------- Model Setup --------------------------------------

    # define number of classes in dataset
    num_classes = 2

    # define class names
    class_names = [0, 1]

    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) # weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # Load previously trained model
    try:
        print("Searching for previously trained available models...")
        model_nr, folder = latest_model(application=model_save_path)
        save_path = folder.joinpath('model'+str(model_nr)+'.pt')
        model.load_state_dict(torch.load(save_path))
        print("model", str(model_nr), "loaded successfully!")
    except FileNotFoundError:
        print("No previously trained models available")
        pass

    # load pretrained model
    # model.load_state_dict(torch.load('models/model12'))

    dataset_val = ChipCorrosionDataset(root_val, transforms=torchvision.transforms.ToTensor()) # get_transform(train=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, collate_fn=lambda x:list(zip(*x)))

    # Look at some images and predicted bbox's after training
    images, targets = next(iter(data_loader_val))
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    exp_empty_tensor = targets[0]['boxes'].size(dim=0) == 0

    model = model.double()
    model.eval()
    with torch.no_grad():
        output = model(images)

    pred_empty_tensor = output[0]['boxes'].size(dim=0) == 0
    if pred_empty_tensor:
        print('\n' + 'No corrosion detected')
        print('Rating = 0')
        pass
    else:
        true_cor_area, pred_cor_area, num_masks, max_mask_size = view_mask(targets, images, output)


        # print('\n' + "Total image area: ", image_area, "pixels")
        # print("Total filter paper area: ", filter_area, "pixels")
        #
        # true_percent_defect = round((true_cor_area / filter_area * 100), 2)
        # pred_percent_defect = round((corrosion_area / filter_area * 100), 2)
        #
        # # get pixel diamter based on filter paper area
        # pixel_diameter = 2 * sqrt(filter_area/pi)
        # # get number of pixel per mm
        # pixel_per_mm = pixel_diameter / 40
        #
        # # get max defect diameter in mm
        # max_mask_mm = round(max_mask_size / pixel_per_mm, 2)
        #
        #
        # def rating_calc(percent_defect):
        #     if 0 < num_masks <= 3:
        #         if max_mask_size < pixel_per_mm:
        #             rating = 1
        #         else:
        #             rating = 2
        #     elif num_masks > 3 and percent_defect < 1:
        #         rating = 2
        #     elif 1 < percent_defect < 5:
        #         rating = 3
        #     elif percent_defect > 5:
        #         rating = 4
        #     else:
        #         rating = 'Invalid'
        #     return rating
        #
        # true_rating = rating_calc(true_percent_defect)
        # pred_rating = rating_calc(pred_percent_defect)
        #
        # diff = round(pred_percent_defect - true_percent_defect, 2)
        # print('\n' + "TRUE corrosion area: ", true_cor_area, "pixels")
        # print("TRUE corrosion percent: ", true_percent_defect, '%')
        # print("TRUE corrosion rating: ", true_rating)
        # print('\n' + "PREDICTED number of defects: ", num_masks)
        # print("PREDICTED single defect max diameter: ", max_mask_mm, 'mm')
        # print('\n' + "PREDICTED corrosion area: ", corrosion_area, "pixels")
        # print("PREDICTED corrosion percent: ", pred_percent_defect, '%')
        # print("PREDICTED corrosion rating: ", pred_rating)
        # print('\n' + 'PREDICTED vs TRUE Difference: ', diff, '%')
        #
        # # num = 0
        # # sleep(1)
        # # for images, targets in data_loader_val:
        # #     if num < 1:
        # #         images = list(image.to(device) for image in images)
        # #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # #
        # #         model = model.double()
        # #         model.eval()
        # #         output = model(images)
        # #
        # #         num += 1
        # #         try:
        # #             target_im = targets[0]['masks'][0].cpu().detach().numpy()
        # #
        # #             # torchvision.utils.make_grid(images[i])
        # #             for i in range(1):
        # #                 # out = output[i]['scores'].to('cpu')
        # #                 # out = out.detach().numpy()
        # #                 for j in range(len(output[i]['scores'])):
        # #                     if j < 0.7:
        # #                         output[i]['boxes'][j] = torch.Tensor([0, 0, 0, 0])
        # #
        # #             # Plot target (true) masks
        # #             for k in range(len(targets[0]['masks'])):
        # #                 target_im2 = targets[0]['masks'][k].cpu().detach().numpy()
        # #                 target_im2[target_im2 > 0.5] = 1
        # #                 target_im2[target_im2 < 0.5] = 0
        # #                 target_im = target_im + target_im2
        # #
        # #             target_im[target_im > 0.5] = 1
        # #             target_im[target_im < 0.5] = 0
        # #             target_im = target_im.astype('int64')
        # #
        # #             # Plot output (predicted) masks
        # #             output_im = output[0]['masks'][0][0, :, :].cpu().detach().numpy()
        # #             for k in range(len(output[0]['masks'])):
        # #                 output_im2 = output[0]['masks'][k][0, :, :].cpu().detach().numpy()
        # #                 output_im2[output_im2 > 0.5] = 1
        # #                 output_im2[output_im2 < 0.5] = 0
        # #                 output_im = output_im + output_im2
        # #
        # #             output_im[output_im > 0.5] = 1
        # #             output_im[output_im < 0.5] = 0
        # #             output_im = output_im.astype('int64')
        # #
        # #             dice_coef_score = dice_coef(y_real=target_im, y_pred=output_im)
        # #             IoU_score = IoU(y_real=target_im, y_pred=output_im)
        # #             # f1_score = get_f1_score(target_im, output_im)
        # #
        # #             print('\n' + 'IoU Score:', round(IoU_score, 3))
        # #             print('Dice Coefficient:', round(dice_coef_score, 3))
        # #             # print('Image f1 score for test set:', f1_score)
        # #
        # #         except IndexError as e:
        # #             print(e)
        # #             pass
        # #
        # #     else:
        # #         del num
        # #         break
