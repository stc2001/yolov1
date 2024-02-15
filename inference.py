import torch
from torchvision import transforms
from model import Yolov1
from PIL import Image
from utils import convert_predboxes , cellboxes_to_boxes
import cv2
from metrics import nms
#import matplotlib.pyplot as plt
split_size=7
num_boxes=2
num_classes=10
ob = []
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_PATH = "/home/gexingdeyun/文档/深度学习/YOLOLearn/yolov1-main/weights_epoch195.pth"  # 替换为实际的权重文件路径

def inference_fn(model, image_path):
    #model.eval()

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)

    return output

def index_of_max_value(lst):
    return max(range(len(lst)), key=lst.__getitem__)

def return_max_result(output):
    temp=[]
    for i in range(len(output)):
        percentage = output[i][1]
        temp.append(percentage)

    index = index_of_max_value(temp)
    result = output[index]
    return result

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=10).to(DEVICE)

    # 加载预训练的权重文件
    model.load_state_dict(torch.load(WEIGHT_PATH))

    image_path = "/home/gexingdeyun/文档/深度学习/datasets/MNISTDetectionDatasetYOLO/images/val2017/5_8563.png"  # 替换为实际的图像文件路径
    image = cv2.imread(image_path)
    height,width = image.shape[:2]
    output = inference_fn(model, image_path)

    #output = output.reshape(-1, split_size, split_size, (num_boxes * 5 + num_classes))
    # 处理输出结果
    output= convert_predboxes(output)
    output = cellboxes_to_boxes(output)
    output = nms(output[0], iou_threshold=0.25, conf=0.25)
    #print(output)
    result = return_max_result(output)


    name , pred , x_centre , y_centre , w , h = result
    xmin = int(x_centre * width - w * width / 2)        # 坐标转换
    ymin = int(y_centre * height - h * height / 2)
    xmax = int(x_centre * width + w * width / 2)
    ymax = int(y_centre * height + h * height / 2)

    tmp = [name, xmin, ymin, xmax, ymax]  # 单个检测框
    ob.append(tmp)

    # 绘制检测框
    for name, x1, y1, x2, y2 in ob:
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)  # 绘制矩形框
        cv2.putText(image, str(name), (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, thickness=1, color=(0, 0, 255))

        # 保存图像

    cv2.imwrite('result.png', image)

    #plt.imshow(image)
    #plt.show()
    print(result)
    #print(output[0])

if __name__ == "__main__":
    main()
