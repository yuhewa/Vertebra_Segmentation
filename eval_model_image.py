import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import engine
import math
import sys
import time
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
import transforms as T

from cv2 import cv2
import os

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

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

# 圖片並排顯示
def multishow(img, label, pred):
    fig=plt.figure(figsize=(8, 5))
    fig.add_subplot(1, 3, 1)
    plt.imshow(img, cmap ='gray')
    fig.add_subplot(1, 3, 2)
    plt.imshow(label, cmap ='gray') # 1 - label 反轉黑白
    fig.add_subplot(1, 3, 3)
    plt.imshow(pred, cmap ='gray')
    plt.show()

def dice_coeff(true_mask, pred_mask, non_seg_score=1.0):
    assert true_mask.shape == pred_mask.shape
    # 化為布林矩陣
    true_mask = np.asarray(true_mask).astype(np.bool)
    pred_mask = np.asarray(pred_mask).astype(np.bool)

    # 若兩者皆為0, 無法算出分數
    img_sum = true_mask.sum() + pred_mask.sum()
    if img_sum == 0:
        return non_seg_score

    # 計算dice coeff
    intersection = np.logical_and(true_mask, pred_mask)
    return 2. * intersection.sum() / img_sum


if __name__ == "__main__":
    
    root_dir = os.getcwd()
    test_dir = 'f01'
    final_test = 'Final_test'
    
    # # normal test
    # test_img_dir = os.path.join(root_dir, "vertebral_processed",test_dir)
    # output_dir = os.path.join(root_dir, "vertebral_processed","final_f01f02f03","output")
    
    # final test
    test_img_dir = os.path.join(root_dir, final_test)
    output_dir = os.path.join(root_dir,final_test,'output')
    
    # 欲使用的model位置
    # modelfile_path = os.path.join(root_dir,'saved_model','first_fine_model.pth')
    modelfile_path = os.path.join(root_dir,'saved_model','f01f02f03blur_model.pth')
    
    # 定義欲使用model與載入參數
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load(modelfile_path))
    
    dataset = imageDataset(test_img_dir, get_transform(train=False))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=utils.collate_fn)
    
    model.eval()
    torch.no_grad()
    
    # # ------------------------------------------------------------------------------
    # # normal test
    # # 因為原本的mask已經處理過了, 需要從其他地方取出原本的mask
    # test_mask_dir = os.path.join(root_dir, 'vertebral',test_dir,'label')
    # test_mask_name =os.listdir(os.path.join(root_dir, 'vertebral',test_dir,'label')) #string array
    # test_masks = np.zeros((len(test_mask_name), 1200, 500), dtype=np.bool)
    # for i in range(len(test_mask_name)):
    #     test_mask_path = os.path.join(test_mask_dir, test_mask_name[i])
    #     test_masks[i] = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)

    # ------------------------------------------------------------------------------
    # final test
    final_test_mask_dir = os.path.join(root_dir, final_test, 'label')
    final_test_mask_name =os.listdir(final_test_mask_dir) #string array

    final_test_masks = np.zeros((len(final_test_mask_name), 1200, 500), dtype=np.bool)
    for i in range(len(final_test_mask_name)):
        final_test_mask_path = os.path.join(final_test_mask_dir, final_test_mask_name[i])
        final_test_masks[i] = cv2.imread(final_test_mask_path, cv2.IMREAD_GRAYSCALE)


    print()
    print()

    # Threshold_val = 0.76
    Threshold_val = 0.5
    threshold = torch.tensor([Threshold_val])
    
    total_dice = 0
    file_index = 1
    for img, target in data_loader:
        
        pred = model(img)
        pred = pred[0] # 因為外面包了一層list, 故取出

        # # boxes labels scores masks
        # print(pred['boxes'].shape)
        # print(pred['labels'].shape)
        # print(pred['scores'].shape)
        # print(pred['masks'].shape)
        
        
        masks = pred['masks']
        # np_mask = pred['masks'].detach().numpy()

        masks = masks.cpu() > threshold.cpu()
        masks = masks.numpy().squeeze()
        
        
        
        region_number = pred['scores'].shape[0]
        vertebra_number = 1

        mask = masks[0] # 新增mask, 等於masks[i]是除了一樣大小, 也事先將mask[0]算入
        for i in range(region_number-1):
            if (pred['scores'][i+1] > 0.67 ): # 0.81
                mask += masks[i+1]
                vertebra_number += 1
            # mask += masks[i+1]

        # 計算dice 下列二擇一
        # test_masks[file_index-1]
        # final_test_masks[file_index-1]
        dice = dice_coeff( final_test_masks[file_index-1] , mask)
        total_dice += dice
        


        # 列印評測訊息
        print()
        print('file',file_index)
        print('region_number:   ', region_number)
        print('vertebra_number: ', vertebra_number)
        print('dice score: ',dice)
        # print('scores: ', pred['scores'][0:vertebra_number])
        # print('not use scores: ', pred['scores'][vertebra_number:20])
        print()



        # # 將圖片結果寫入並存下來
        # mask = np.array(mask, dtype=np.uint8)
        # mask = mask * 255
        # name = 'report_f01f02f03_'+str(file_index) + '.png'
        # cv2.imwrite(os.path.join(output_dir,name) , mask)


        file_index += 1
    
    print()
    print('--------------------------------------')
    print('avg dice: ',total_dice / (file_index-1) )
    print('--------------------------------------')

