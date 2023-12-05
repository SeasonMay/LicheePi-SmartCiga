import torch
from torch.nn import functional
from torch import nn
from PIL import Image
from torchvision import transforms, datasets
from vgg_train import vgg16


class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
               '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
               '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48']


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 载入模型并读取权重
model = vgg16
model.load_state_dict(torch.load("save_model_/last_model.pth"))
model.cuda()
model.eval()

img_path = r'E:\vgg_ciga\val\48\钻石（软荷花）.jpg'  # 测试图片路径

transforms_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 图像标准化处理
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open(img_path)
img_ = transforms_valid(img).unsqueeze(0)

img_ = img_.to(device)
outputs = model(img_)


# 输出概率最大的类别
_, indices = torch.max(outputs, 1)
percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
perc = percentage[int(indices)].item()
result = class_names[indices]
print('predicted:', result)


# 得到预测结果，并且从大到小排序
_, indices = torch.sort(outputs, descending=True)
# 返回每个预测值的百分数
percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
print([(class_names[idx], percentage[idx].item()) for idx in indices[0][:5]])
