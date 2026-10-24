智慧林业-无人机航拍丛林树种检测数据集，4604张，yolo，voc，coco三种标注方式
图像尺寸:640*640![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/58f7cdeba816490cbb287c7dd344f549.png)

类别数量:3类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5ff2a85b7c0946009f064607d14fb176.png)

训练集图像数量:3220; 验证集图像数量:692； 测试集图像数量:692
类别名称: 每一类图像数 ，每一类标注数![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d41c0ecefefc4faabc5069af0f9d5230.png)

Birch-桦树：3426,41129
Pine-松树：4318,124522
Spruce-云杉：4409,121848
image num: 4604
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5ea807be7f884ae6b4c4b670a07e796c.png)


模型代码
采用 YOLOv11n 网络训练
训练轮次：80 个 epoch![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dd52a3385bd345eab76992296b7a9498.png)

提供全部训练 + 测试源代码
训练精度 mAP 效果如图所示![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3880a2c15a8e457caee00ee240b7abbe.png)

PyQt5 界面功能![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6abe9b70d6534b8fa9db58c3bdbf0f49.png)

界面使用 PyQt5 开发，提供全部源码（.ui、.qrc、.py 及图标文件）
支持图片检测、视频检测、摄像头实时检测
界面实时显示：目标位置、目标总数、置信度等信息
支持检测结果保存导出![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6d0ae6acc11e4496868c292ca3f75983.png)

操作简单直观，无需命令行
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/272e596711e34f38b3fad99870a48082.png)


| 项目 | 详情 |
| --- | --- |
| 数据集名称 | 航拍丛林树种检测数据集 |
| 图像总量 | 4604张 |
| 图像分辨率 | 640×640 |
| 标注格式 | YOLO、VOC、COCO |
| 数据集划分 | 训练集：3220张；验证集：692张；测试集：692张 |
| 类别总数 | 3类 |
| 类别明细 | Birch-桦树：3426张图像，41129个标注   Pine-松树：4318张图像，124522个标注   Spruce-云杉：4409张图像，121848个标注 |
| 训练模型 | YOLOv11n |
| 训练轮次 | 80 epoch |
| 配套交付 | 训练+测试全套源码、PyQt5可视化界面全套源码（.ui/.qrc/.py/图标） |
| 界面功能 | 图片检测、视频检测、摄像头实时检测；显示目标框、目标数量、置信度；检测结果导出保存，图形化操作，无需命令行 |

## 一、环境安装

```
pip install ultralytics pyqt5 opencv-python pillow
```

## 二、数据集配置文件 trees.yaml

```
path: ./trees_dataset
train: train/images
val: val/images
test: test/images
names:
  0: Birch
  1: Pine
  2: Spruce
```

## 三、训练脚本 train.py

```
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="trees.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        project="runs/train",
        name="tree_detect"
    )
```

## 四、PyQt5 推理界面 main.py

```
import sys, cv2, os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ultralytics import YOLO

class TreeDetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("航拍丛林树种检测系统 YOLOv11n")
        self.resize(1100,750)
        self.model = YOLO("./runs/train/tree_detect/weights/best.pt")
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.camera_frame)
        self.init_ui()

    def init_ui(self):
        w = QWidget()
        self.setCentralWidget(w)
        lay = QVBoxLayout(w)
        btn_lay = QHBoxLayout()
        self.btn_img = QPushButton("图片检测")
        self.btn_vid = QPushButton("视频检测")
        self.btn_cam = QPushButton("摄像头实时")
        self.btn_save = QPushButton("保存结果")
        self.info_label = QLabel("目标总数：0")
        btn_lay.addWidget(self.btn_img)
        btn_lay.addWidget(self.btn_vid)
        btn_lay.addWidget(self.btn_cam)
        btn_lay.addWidget(self.btn_save)
        btn_lay.addWidget(self.info_label)
        self.label_show = QLabel()
        self.label_show.setAlignment(Qt.AlignCenter)
        lay.addLayout(btn_lay)
        lay.addWidget(self.label_show)

        self.btn_img.clicked.connect(self.detect_image)
        self.btn_vid.clicked.connect(self.detect_video)
        self.btn_cam.clicked.connect(self.open_camera)
        self.btn_save.clicked.connect(self.save_result)
        self.result_img = None

    def detect_image(self):
        fp,_ = QFileDialog.getOpenFileName(self,"选择图片","","Image(*.jpg *.png *.jpeg)")
        if not fp:return
        res = self.model(fp)[0]
        img = res.plot()
        cnt = len(res.boxes)
        self.info_label.setText(f"目标总数：{cnt}")
        self.result_img = img
        self.show_img(img)

    def detect_video(self):
        fp,_ = QFileDialog.getOpenFileName(self,"选择视频","","Video(*.mp4 *.avi)")
        if not fp:return
        self.capture = cv2.VideoCapture(fp)
        self.timer.start(30)

    def open_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.timer.start(30)

    def camera_frame(self):
        ret,frame = self.capture.read()
        if not ret:
            self.timer.stop()
            return
        res = self.model(frame)[0]
        img = res.plot()
        cnt = len(res.boxes)
        self.info_label.setText(f"目标总数：{cnt}")
        self.result_img = img
        self.show_img(img)

    def show_img(self,img):
        rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,c = rgb.shape
        qimg = QImage(rgb.data,w,h,w*c,QImage.Format_RGB888)
        self.label_show.setPixmap(QPixmap.fromImage(qimg).scaled(self.label_show.size(),Qt.KeepAspectRatio))

    def save_result(self):
        if self.result_img is None:
            QMessageBox.warning(self,"提示","暂无检测结果！")
            return
        fp,_ = QFileDialog.getSaveFileName(self,"保存图片","res.jpg","Image(*.jpg)")
        if fp:
            cv2.imwrite(fp,self.result_img)
            QMessageBox.information(self,"成功","已保存！")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = TreeDetectWindow()
    win.show()
    sys.exit(app.exec_())
```

## 五、总结

航拍丛林树种检测数据集，合计4604张640×640航拍图像，支持YOLO、VOC、COCO三种标注格式，包含桦树、松树、云杉3类树种，标注样本量充足。基于YOLOv11n训练，80轮epoch，配套完整训练测试代码与PyQt5图形化检测界面，支持图片、视频、摄像头实时识别、结果导出，适用于森林资源调查、林地树种分类、林业遥感监测、植被普查等项目开发。

### 应用价值标签

✅ 航拍森林树种自动识别
✅ 林业资源调查与植被普查
✅ 森林蓄积量估算辅助
✅ 遥感小目标树种检测算法验证
✅ 森林防火、林地生态监测系统开发


