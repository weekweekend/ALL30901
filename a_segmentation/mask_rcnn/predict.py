import os
import time
import json
from skimage import io
# file_path = 'results/mask{:d}.jpg'.format(num)
# io.imsave(os.path.join('results/mask', file), out)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs
from torchvision.utils import save_image

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    num_classes = 6  # 不包含背景
    box_thresh = 0.5
    weights_path = "./save_weights/model_800.pth"
    img_path = "testData/2_2.png"
    imgs_path = "testData"

    label_json_path = './coco91_indices.json'
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')["model"])
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    num=0
    for parent, _, files in os.walk(imgs_path):
        files.sort()  # 排序一下
        for file in files:                
            pic_path = os.path.join(parent, file)
            # img = cv2.imread(pic_path)
            print(pic_path)

            # load image
            assert os.path.exists(pic_path), f"{pic_path} does not exits."
            original_img = Image.open(pic_path).convert('RGB')

            # from pil image to tensor, do not normalize image
            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            model.eval()  # 进入验证模式
            with torch.no_grad():
                # init
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)

                t_start = time_synchronized()
                predictions = model(img.to(device))[0]
                t_end = time_synchronized()
                print("inference+NMS time: {}".format(t_end - t_start))

                predict_boxes = predictions["boxes"].to("cpu").numpy()
                predict_classes = predictions["labels"].to("cpu").numpy()
                predict_scores = predictions["scores"].to("cpu").numpy()
                predict_mask = predictions["masks"].to("cpu").numpy()
                print(predict_classes)
                predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]
                _,h,w = predict_mask.shape
                if len(predict_boxes) == 0:
                    print("没有检测到任何目标!")
                    return
                
                out = np.zeros((h,w))
                for ii in range(len(predict_classes)):
                    res = predict_mask[ii:ii+1,:,:].reshape((h,w))
                    res[res>=0.7]=1
                    res[res<0.7]=0
                    out = out + res*predict_classes[ii]*10

                print(out.max())
                file_path = 'results/mask{:d}.jpg'.format(num)
                io.imsave(os.path.join('results/mask', file), out)

                plot_img = draw_objs(original_img,
                                    boxes=predict_boxes,
                                    classes=predict_classes,
                                    scores=predict_scores,
                                    masks=predict_mask,
                                    category_index=category_index,
                                    line_thickness=1,
                                    font='arial.ttf',
                                    font_size=20)
                plt.imshow(plot_img)
                plt.show()
                # 保存预测的图片结果
                file_path2 = 'results/test{:d}.jpg'.format(num)
                plot_img.save(os.path.join('results/test', file))
                num = num+1




    # # load image
    # assert os.path.exists(img_path), f"{img_path} does not exits."
    # original_img = Image.open(img_path).convert('RGB')

    # # from pil image to tensor, do not normalize image
    # data_transform = transforms.Compose([transforms.ToTensor()])
    # img = data_transform(original_img)
    # # expand batch dimension
    # img = torch.unsqueeze(img, dim=0)

    # model.eval()  # 进入验证模式
    # with torch.no_grad():
    #     # init
    #     img_height, img_width = img.shape[-2:]
    #     init_img = torch.zeros((1, 3, img_height, img_width), device=device)
    #     model(init_img)

    #     t_start = time_synchronized()
    #     predictions = model(img.to(device))[0]
    #     t_end = time_synchronized()
    #     print("inference+NMS time: {}".format(t_end - t_start))

    #     predict_boxes = predictions["boxes"].to("cpu").numpy()
    #     predict_classes = predictions["labels"].to("cpu").numpy()
    #     predict_scores = predictions["scores"].to("cpu").numpy()
    #     predict_mask = predictions["masks"].to("cpu").numpy()
    #     print(predict_classes)
    #     predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]
    #     _,h,w = predict_mask.shape
    #     if len(predict_boxes) == 0:
    #         print("没有检测到任何目标!")
    #         return
        
    #     out = np.zeros((h,w))
    #     for ii in range(len(predict_classes)):
    #         res = predict_mask[ii:ii+1,:,:].reshape((h,w))
    #         res[res>=0.7]=1
    #         res[res<0.7]=0
    #         out = out + res*predict_classes[ii]*10

    #     print(out.max())
    #     file_path = 'results/test.jpg'
    #     io.imsave(file_path, out)

    #     plot_img = draw_objs(original_img,
    #                          boxes=predict_boxes,
    #                          classes=predict_classes,
    #                          scores=predict_scores,
    #                          masks=predict_mask,
    #                          category_index=category_index,
    #                          line_thickness=1,
    #                          font='arial.ttf',
    #                          font_size=20)
    #     plt.imshow(plot_img)
    #     plt.show()
    #     # 保存预测的图片结果
    #     plot_img.save("results/test_result.jpg")






if __name__ == '__main__':
    main()

