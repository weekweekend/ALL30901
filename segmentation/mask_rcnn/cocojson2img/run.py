# from pycocotools.coco import COCO

# import matplotlib.pyplot as plt
# import cv2

# import os
# import numpy as np
# import random
# cocoRoot = "../data/coco/"
# dataType = "val2017"

# annFile = os.path.join(cocoRoot, f'annotations/instances_{dataType}.json')

# # # initialize COCO api for instance annotations
# coco=COCO(annFile)

# # 利用getCatIds函数获取某个类别对应的ID，
# # 这个函数可以实现更复杂的功能，请参考官方文档
# ids = coco.getCatIds('pipe')[0]
# print(f'"pipe" 对应的序号: {ids}')


# # # 利用loadCats获取序号对应的文字类别
# # # 这个函数可以实现更复杂的功能，请参考官方文档
# # cats = coco.loadCats(1)
# # print(f'"1" 对应的类别名称: {cats}')

# # 获取包含 *类 的所有图片
# id = coco.getCatIds(['pipe'])[0]
# imgIds = coco.catToImgs[id]
# print(f'包含pipe的图片共有: {len(imgIds)}张, 分别是：')
# print(imgIds)

# imgId = imgIds[1]

# imgInfo = coco.loadImgs(imgId)[0]
# print(f'图像{imgId}的信息如下：\n{imgInfo}')
 
# annIds = coco.getAnnIds(imgIds=imgInfo['id'])
# anns = coco.loadAnns(annIds)
# print(f'图像{imgInfo["id"]}包含{len(anns)}个ann对象，分别是:\n{annIds}')

# print(f'ann{annIds[0]}对应的mask如下：')
# mask = coco.annToMask(anns[1])
# plt.imshow(mask); plt.axis('off')
# plt.savefig('res.jpg')



import os
import random

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage import io

random.seed(0)

json_path = "../data/coco76/annotations/instances_val2017.json"

img_path = "../data/coco76/val2017"

# random pallette
pallette = [0, 0, 0] + [random.randint(0, 255) for _ in range(255*3)]

# load coco data
coco = COCO(annotation_file=json_path)

# get all image index info
ids = list(sorted(coco.imgs.keys()))
print("number of images: {}".format(len(ids)))

# get all coco class labels
coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])

# 遍历前三张图像
for img_id in ids[:len(ids)]:
    # 获取对应图像id的所有annotations idx信息
    ann_ids = coco.getAnnIds(imgIds=img_id)
    # 根据annotations idx信息获取所有标注信息
    targets = coco.loadAnns(ann_ids)

    # get image file name
    path = coco.loadImgs(img_id)[0]['file_name']

    # read image
    img = Image.open(os.path.join(img_path, path)).convert('RGB')
    img_w, img_h = img.size

    masks = []
    cats = []
    for target in targets:
        cats.append(target["category_id"])  # get object class id
        polygons = target["segmentation"]   # get object polygons
        rles = coco_mask.frPyObjects(polygons, img_h, img_w)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = mask.any(axis=2)
        masks.append(mask)

    cats = np.array(cats, dtype=np.int32)
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)

    # merge all instance masks into a single segmentation map
    # with its corresponding categories
    target = (masks * cats[:, None, None]).max(axis=0)
    # discard overlapping instances
    print(target.shape) 
    # file_path = 'res{:d}.png'.format(img_id)
    io.imsave(os.path.join('res', path), target)


    target[masks.sum(0) > 1] = 255
    target = Image.fromarray(target.astype(np.uint8))
    target.putpalette(pallette)
    plt.imshow(target)
    plt.show()
    # plt.savefig('res.jpg')