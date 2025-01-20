#!/usr/bin/env python3
from time import sleep
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm
from natsort import natsorted
import warnings
import matplotlib.cbook
from pathlib import Path
import argparse
import os
import re
import glob
import numpy as np



BASE_PATH = Path(__file__).parent.parent


class ChipCorrosionDataset(Dataset):
    def __init__(self, root, transforms=None):  # transforms
        self.root = root

        # self.transforms = transforms
        self.transforms = []
        if transforms != None:
            self.transforms.append(transforms)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(natsorted(os.listdir(os.path.join(root, "images"))))
        # print('imgs file names:', self.imgs)
        self.masks = list(natsorted(os.listdir(os.path.join(root, "masks"))))
        # print('masks file names:', self.masks)
        # images = []
        # masks = []
        # image_names = [f.name for f in os.scandir(root)]
        # image_names = natsorted(image_names)
        # for image_name in image_names:
        #   if image_name.endswith('_anno.bmp'):
        #     masks.append(image_name)
        #   else:
        #     images.append(image_name)
        # print(images)
        # print(masks)
        # print('Length images:', len(images))
        # print('Length masks:', len(masks))
        # self.imgs = images
        # self.masks = masks
        # print('Length images:', len(self.imgs))
        # print('Length masks:', len(self.masks))

    def __getitem__(self, idx):
        # idx sometimes goes over the nr of training images, add logic to keep it lower
        if idx >= 80:
            idx = np.random.randint(80, size=1)[0]
        # print("idx:", idx)

        # load images ad masks
        # print('idx:', idx)
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        # print(img_path)
        # print('img_path', self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        # print(mask_path)
        # print('mask_path', self.masks[idx])
        # print('img_path name:', self.imgs[idx])
        # img_path = os.path.join(self.root, self.imgs[idx])
        # print('mask_path name:', self.masks[idx])
        # mask_path = os.path.join(self.root, self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB, because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path).convert("L")
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        # print(num_objs)
        boxes = []
        # print(num_objs)
        for i in range(num_objs):
            pos = np.where(masks[i])
            # print('pos', pos)
            # print(pos)
            xmin = np.min(pos[1] - 1)
            # print("xmin:", xmin)
            xmax = np.max(pos[1] + 1)
            # print("xmax:", xmax)
            ymin = np.min(pos[0] - 1)
            # print("ymin:", ymin)
            ymax = np.max(pos[0] + 1)
            # print("ymax:", ymax)
            # print('box: ', [xmin, ymin, xmax, ymax])
            # Check if area is larger than a threshold
            # A = abs((xmax - xmin) * (ymax - ymin))
            # print('x', (xmax - xmin))
            # print('y', (ymax - ymin))
            # print('area', A)
            # if A < 9:
            #     # print('Nr before deletion:', num_objs)
            #     # obj_ids = np.delete(obj_ids, [i])
            #     # # print('Area smaller than 5! Box coordinates:', [xmin, ymin, xmax, ymax])
            #     # print('Nr after deletion:', len(obj_ids))
            #     # continue
            #     # xmax=xmax+5
            #     # ymax=ymax+5
            #     pass
            #     print('PASSED')
            # else:
            boxes.append([xmin, ymin, xmax, ymax])

        # print('nr boxes is equal to nr ids:', len(boxes)==len(obj_ids))
        num_objs = len(obj_ids)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # # suppose all instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        for i in self.transforms:
            img = i(img)

        target = {}

        if len(boxes) == 0:
            boxes = torch.zeros(1, 4)
            boxes[2] = img.size[1]
            boxes[3] = img.size[2]
        if len(labels) == 0:
            labels = torch.zeros(1)
        if len(masks) == 0:
            masks = torch.zeros(img.shape)

        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["labels"] = labels  # Not sure if this is needed

        # trans = transforms.ToTensor()
        # img = trans(img)
        # transform = []
        # transform.append(transforms.RandomHorizontalFlip(0.5))
        # transform.append(transforms.ToTensor())
        # transform = Compose(transform)
        # img, target = transform(img, target)

        # if self.transforms is not None:
        #   # for t in self.transforms:
        #   #   img = t(img)
        #   #   target = t(target)
        #     # print(img.size)
        #     img, target = self.transforms(img, target)

        return img.double(), target

    def __len__(self):
        return len(self.imgs)


def latest_model(application):
    sp = application
    largest = 0
    model_names = os.listdir(os.path.join(sp))
    for i in model_names:
        nr = int(re.findall(r'\d+', i)[0])
        # print(nr)
        if nr > largest:
            largest = nr
    return largest, Path(sp)


if __name__ == '__main__':
    # ------------------------------- Run Model ----------------------------------------
    # set the following 3 variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', dest='train_dir', action='store', help='Path to the Training Data')
    parser.add_argument('--mrcnn_path', dest='checkpoints_dir', action='store', help='Path to the Saved Models')
    parser.add_argument('--val_dir', dest='val_dir', action='store', help='Path to the Validation Data')
    parser.add_argument('--n_epochs', dest='n_epochs', action='store', help='Number of Epochs')
    parser.add_argument('--name', dest='name', action='store', help='Name of CV Application')

    arguments = parser.parse_args()

    n_epochs = int(arguments.n_epochs)
    train_folder = arguments.train_dir     # 'train80000'
    test_folder = arguments.val_dir        #'test20000'
    model_application = arguments.name
    model_save_path = arguments.checkpoints_dir
    # ------------------------------ Generic Setup --------------------------------------

    # ignore warnings
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


    # Check if GPU is available
    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    # ----------------------------- Training & Model Setup ------------------------------------

    # ---------------------- Initial dataset and dataloader setup --------------------------

    # get root paths for train and test data
    root_train = train_folder
    root_test = test_folder

    # define training set class
    dataset_train = ChipCorrosionDataset(root_train, transforms=torchvision.transforms.ToTensor()) # get_transform(train=True)

    # Define data loader for training set
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, collate_fn=lambda x: list(zip(*x)))

    # # View some images and their bbox's in the training set
    # images, labels = next(iter(data_loader_train))
    # view(images=images, labels=labels, n=2, std=1, mean=0)

    # ------------------------------- Model Setup --------------------------------------

    # define number of classes in dataset
    num_classes = 2

    # load an instance segmentation model pre-trained on COCO
    # choose from maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
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

    # ------------------------------- Training --------------------------------------
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

    sleep(1)
    # Perform training loop for n epochs
    loss_list = []
    model.train()
    for epoch in tqdm(range(n_epochs)):
        loss_epoch = []
        iteration = 1
        for images, targets in tqdm(data_loader_train):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            model = model.double()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            loss_epoch.append(losses.item())

            iteration += 1

        loss_epoch_mean = np.mean(loss_epoch)
        loss_list.append(loss_epoch_mean)
        sleep(1)
        print("Average loss for epoch = {:.4f} ".format(loss_epoch_mean))
        sleep(1)

        ## ADD IN SAVING OF MODEL SCALED TO TRAINING DURATION TO ALLOW FOR INTERMEDIATE CHECKPOINTS

    # Save the trained model to the 'models' folder
    model_nr, folder = latest_model(application=model_save_path)
    save_path = folder.joinpath('model' + str(model_nr+1) + '.pt')
    torch.save(model.state_dict(), save_path)

# Plot training loss
# plt.plot(list(range(n_epochs)), loss_list, label='traning loss')
# plt.legend()
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# # plt.show()
# plt.savefig('/home/rraithel/drv1/pythonProjects/ChipCorrosionMRCNN/results/training loss.png')

# ------------------------------- Testing --------------------------------------

# dataset_test = ChipCorrosionDataset(root_test, transforms=torchvision.transforms.ToTensor()) # get_transform(train=True)
# data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, collate_fn=lambda x:list(zip(*x)))
#
# # Look at some images and predicted bbox's after training
# images, targets = next(iter(data_loader_test))
# images = list(image.to(device) for image in images)
# targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
# model = model.double()
# model.eval()
# output = model(images)

# with torch.no_grad():
#     view(images, output, 1)

# torchvision.utils.make_grid(images[i])
# for i in range(1):
#     # out = output[i]['scores'].to('cpu')
#     # out = out.detach().numpy()
#     for j in range(len(output[i]['scores'])):
#         if j < 0.7:
#             output[i]['boxes'][j] = torch.Tensor([0, 0, 0, 0])

# View a sample of masks predicted on test data
# view_mask(targets, output, n=1)

# ------------------------------- Results --------------------------------------

# Get IoU score for whole test set
# IoU_scores_list = []
# dice_coef_scores_list = []
# f1_scores_list = []
# skipped = 0
# for images, targets in tqdm(data_loader_test):
#
#     images = list(image.to(device) for image in images)
#     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#     model = model.double()
#     model.eval()
#     output = model(images)
#     target_im = targets[0]['masks'][0].cpu().detach().numpy()
#     for k in range(len(targets[0]['masks'])):
#         target_im2 = targets[0]['masks'][k].cpu().detach().numpy()
#         target_im2[target_im2 > 0.5] = 1
#         target_im2[target_im2 < 0.5] = 0
#         target_im = target_im + target_im2
#
#     target_im[target_im > 0.5] = 1
#     target_im[target_im < 0.5] = 0
#     target_im = target_im.astype('int64')
#
#     # Plot output (predicted) masks
#     output_im = output[0]['masks'][0][0, :, :].cpu().detach().numpy()
#     # print(len(output[0]['masks']))
#     for k in range(len(output[0]['masks'])):
#         output_im2 = output[0]['masks'][k][0, :, :].cpu().detach().numpy()
#         output_im2[output_im2 > 0.5] = 1
#         output_im2[output_im2 < 0.5] = 0
#         output_im = output_im + output_im2
#
#     output_im[output_im > 0.5] = 1
#     output_im[output_im < 0.5] = 0
#     output_im = output_im.astype('int64')
#
#     if target_im.shape != output_im.shape:
#         skipped += 1
#         continue
#
#     dice_coef_score = dice_coef(y_real=target_im, y_pred=output_im)
#     dice_coef_scores_list.append(dice_coef_score)
#     IoU_score = IoU(y_real=target_im, y_pred=output_im)
#     IoU_scores_list.append(IoU_score)
#     f1_score = get_f1_score(target_im, output_im)
#     f1_scores_list.append(f1_score)
#
# print('----------------- STATISTICS -----------------')
# print('mean IoU score for test set:', np.mean(IoU_scores_list))
# print('mean Dice Coefficient score for test set:', np.mean(dice_coef_scores_list))
# print('mean f1 score for test set:', np.mean(f1_scores_list))
