# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T


from cv2 import cv2


class imageDataset(object):
    def __init__(self, file_dir, transforms = None):

        self.file_dir =  file_dir
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        self.imgs = list(sorted(os.listdir(os.path.join(file_dir, "image"))))
        self.masks = list(sorted(os.listdir(os.path.join(file_dir, "label"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.file_dir, "image", self.imgs[idx])
        mask_path = os.path.join(self.file_dir, "label", self.masks[idx])

        
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        
        # np.set_printoptions(threshold=np.inf) # 觀看完整array
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)

        img = np.array(img)
        blur = cv2.GaussianBlur(img, (0,0), 3)
        img = blur
        # img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)


        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        

        # split the color-encoded mask into a set of binary masks

        # masks = mask == obj_ids[:, None, None]
        masks = masks = np.zeros( (np.max(mask), mask.shape[0], mask.shape[1] ), dtype=np.uint8)
        # 取出各個物件的mask
        num_objs = len(obj_ids)
        for k in range(num_objs):
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if (mask[i][j] == obj_ids[k]):
                        masks[k][i][j] = True

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

# 若為train則多做一道augmentation
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():

    # 自訂路徑
    root_dir = os.getcwd()


    train_dir = os.path.join(root_dir, "vertebral_processed","final_f01f02f03")

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and verterbra
    num_classes = 2
    # use our dataset and defined transformations
    dataset = imageDataset(train_dir, get_transform(train=True))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True,
        collate_fn=utils.collate_fn)
    # , num_workers=4

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 1 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)\

        # update the learning rate
        lr_scheduler.step()

        # 原本使用的evaluate, 取代成其他方法
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)


    torch.save(model.state_dict(), './saved_model/f01f02f03blur_model.pth')

    print("That's it!")
    
if __name__ == "__main__":
    main()
