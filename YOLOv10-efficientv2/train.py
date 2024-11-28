# coding:utf-8
# By CSDN 小哥谈
from ultralytics import YOLOv10
# 模型配置文件
model_yaml_path = "efficientnetv2.yaml"
# 数据集配置文件
data_yaml_path = 'ultralytics/cfg/datasets/coco.yaml'
# 预训练模型
pre_model_name = 'yolov10n.pt'

if __name__ == '__main__':
 # 加载预训练模型
 model = YOLOv10("efficientnetv2.yaml").load('yolov10n.pt')
 # 训练模型
 results = model.train(data=data_yaml_path,epochs=300,batch=8,name='train_v10')
 # Evaluate the model's performance on the validation set
 results = model.val()