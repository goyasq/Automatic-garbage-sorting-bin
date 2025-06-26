##两个#代表改过的
#存放位置/home/pi/tflite

#导入各类库
from gpiozero import Button, LED, PWMLED
from picamera import PiCamera
from time import sleep

import RPi.GPIO as GPIO
import serial
import time
import math
import smbus

import threading

from PIL import Image##
import cv2##
import numpy as np##
import argparse##
import importlib.util##
import os##

ser = serial.Serial("/dev/ttyAMA0", 9600)
#设置输入输出口
#输入口GPIO 21，接按钮或传感器，检测到电平后，开始启动视觉识别，检测输出垃圾种类

KEY = 21
FULL1 = 20
FULL2 = 16
FULL3 = 24
FULL4 = 23
timeout = 0
VideoFlag=0

camera = PiCamera()

# ============================================================================
# Raspi PCA9685 16-Channel PWM Servo Driver
# ============================================================================

#全局变量定义
HazardousWaste = 0  #有害垃圾
OtherWaste     = 0  #其他垃圾
RecyclableWaste= 0  #可回收垃圾
KitchenWaste   = 0  #厨余垃圾
Battery        = 0
BrickAndTileCeramics = 0
cigarette      = 0
Cans           = 0
WaterBottle    = 0
Fruits         = 0
Vegetables     = 0
i = 0  #序号

#垃圾桶满载标志
full1flag = 0
full2flag = 0
full3flag = 0
full4flag = 0

#垃圾桶满载计时
full1timeout = 0
full2timeout = 0
full3timeout = 0
full4timeout = 0

class PCA9685:

  # Registers/etc.
  __SUBADR1            = 0x02
  __SUBADR2            = 0x03
  __SUBADR3            = 0x04
  __MODE1              = 0x00
  __PRESCALE           = 0xFE
  __LED0_ON_L          = 0x06
  __LED0_ON_H          = 0x07
  __LED0_OFF_L         = 0x08
  __LED0_OFF_H         = 0x09
  __ALLLED_ON_L        = 0xFA
  __ALLLED_ON_H        = 0xFB
  __ALLLED_OFF_L       = 0xFC
  __ALLLED_OFF_H       = 0xFD

  def __init__(self, address=0x40, debug=False):
    self.bus = smbus.SMBus(1)
    self.address = address
    self.debug = debug
    if (self.debug):
      print("Reseting PCA9685")
    self.write(self.__MODE1, 0x00)
	
  def write(self, reg, value):
    "Writes an 8-bit value to the specified register/address"
    self.bus.write_byte_data(self.address, reg, value)
    if (self.debug):
      print("I2C: Write 0x%02X to register 0x%02X" % (value, reg))
	  
  def read(self, reg):
    "Read an unsigned byte from the I2C device"
    result = self.bus.read_byte_data(self.address, reg)
    if (self.debug):
      print("I2C: Device 0x%02X returned 0x%02X from reg 0x%02X" % (self.address, result & 0xFF, reg))
    return result
	
  def setPWMFreq(self, freq):
    "Sets the PWM frequency"
    prescaleval = 25000000.0    # 25MHz
    prescaleval /= 4096.0       # 12-bit
    prescaleval /= float(freq)
    prescaleval -= 1.0
    if (self.debug):
      print("Setting PWM frequency to %d Hz" % freq)
      print("Estimated pre-scale: %d" % prescaleval)
    prescale = math.floor(prescaleval + 0.5)
    if (self.debug):
      print("Final pre-scale: %d" % prescale)

    oldmode = self.read(self.__MODE1)
    newmode = (oldmode & 0x7F) | 0x10        # sleep
    self.write(self.__MODE1, newmode)        # go to sleep
    self.write(self.__PRESCALE, int(math.floor(prescale)))
    self.write(self.__MODE1, oldmode)
    time.sleep(0.005)
    self.write(self.__MODE1, oldmode | 0x80)

  def setPWM(self, channel, on, off):
    "Sets a single PWM channel"
    self.write(self.__LED0_ON_L+4*channel, on & 0xFF)
    self.write(self.__LED0_ON_H+4*channel, on >> 8)
    self.write(self.__LED0_OFF_L+4*channel, off & 0xFF)
    self.write(self.__LED0_OFF_H+4*channel, off >> 8)
    if (self.debug):
      print("channel: %d  LED_ON: %d LED_OFF: %d" % (channel,on,off))
	  
  def setServoPulse(self, channel, pulse):
    "Sets the Servo Pulse,The PWM frequency must be 50HZ"
    pulse = pulse*4096/20000        #PWM frequency is 50HZ,the period is 20000us
    self.setPWM(channel, 0, int(pulse))
    
## 定义和解析模型输入参数
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default='/home/pi/tflite/myModel')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.25)#输出结果的最小置信阈值
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1920x1080')#分辨率
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow库
# 如果安装了tflite_runtime, 从tflite_runtime import interpreter, 否则从常规的tensorflow中import
# 如果使用了Coral Edge TPU, import load_delegate库
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

#如果使用了Edge TPU, 为Edge TPU模型指定文件名
if use_TPU:
    # 如果需要指定.tflite文件名字, 这里需要更改, 否则默认使用'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# 得到当前工作路径
CWD_PATH = os.getcwd()

# .tflite文件的路径, 这个文件包含目标检测模型
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# labelmap.txt的路径
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# 加载label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 加载 Tensorflow Lite 模型
# 如果使用了Edge TPU,使用特殊的load_delegate参数
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

#获取模型细节
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]#输入图片的高
width = input_details[0]['shape'][2]#输入图片的宽

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# 拍照
def take_photo():
    #GPIO 2检测到电平
    
    # 打开相机预览
    camera.start_preview(alpha=200)
    # 等待2s以上，图像自动调节完成，并稳定
    sleep(3) 
    # 相机图像旋转角度
    # 根据需求自己调节
    camera.rotation = 0
    #输出图片的文件路径，一般不需改动
    camera.capture('/home/pi/Pictures/image.jpg')
    ##转换为(1080,1920,3)的nparray
    image=np.array(Image.open('/home/pi/Pictures/image.jpg').convert('RGB'))##
    #相机停止预览
    camera.stop_preview()
    #white_led.off()
    #sleep(1)
    ##返回这个数组
    return image##

#垃圾投入检测 GPIO 21引脚
#检测GPIO21 投入检测传感器电平状态，投入垃圾立即开始识别    
def MyInterrupt(KEY):
    global VideoFlag
    global timeout
    #time.sleep(0.5)  #Key delay times
    while GPIO.input(KEY) == 0: #遮挡投入传感器后，为低电平，等待释放投入传感器的遮挡，防止多次触发识别函数
        if GPIO.input(KEY) == 1: #投入传感器，不再被遮挡，引脚为高电平
            
            VideoFlag = 1 #检测到垃圾投入传感器被遮挡，标志位置1
            timeout=0
            
            ser.write(b'page display')  # 屏幕组件控制指令，信息显示界面
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            ##拍照
            frame1 = take_photo()##
            ##处理图片数组
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)
    
            # 正则化像素值如果使用浮动模型(即如果模型是非量化的)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # 运行检测模型
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # 取得检测结果
            boxes = interpreter.get_tensor(output_details[1]['index'])[0] # 包围框坐标
            classes = interpreter.get_tensor(output_details[3]['index'])[0] # 类索引
            scores = interpreter.get_tensor(output_details[0]['index'])[0] # 置信度
            #num = interpreter.get_tensor(output_details[3]['index'])[0]  # 目标总数 (不准确、非必要)
    have_rubbish = 0
    # 如果置信度高于最小阈值，输出结果
    for i in range(len(scores)):
        if ((scores[i] > 0.4) and (scores[i] <= 1.0)):
            have_rubbish = 1
            # 得到框在画面中的位置
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            #获得类名称
            object_name = labels[int(classes[i])]
            #输出类名称，置信度，左上角坐标，右下角坐标
            print(object_name,scores[i],xmin,ymin,xmax,ymax)
            led_select(object_name)
            #ser.write(('%s %d %d\r\n'%(object_name,xmin,ymin)).encode())#chuankoushuchu
    if have_rubbish == 0:
        led_select('NO_RUBBISH')
GPIO.setmode(GPIO.BCM)
GPIO.setup(KEY,GPIO.IN,GPIO.PUD_UP)
GPIO.add_event_detect(KEY,GPIO.FALLING,MyInterrupt,200) #注意，此处使用下降沿触发方式FALLING，遮挡传感器后为低电平

#满载检测1 GPIO20引脚
def MyInterrupt1(FULL1):
    global full1flag
    global full1timeout
    time.sleep(0.5)  #Key delay times
    if GPIO.input(FULL1) == 1: #Key is low
        full1flag = 1
        full1timeout =0       


GPIO.setmode(GPIO.BCM)
GPIO.setup(FULL1,GPIO.IN,GPIO.PUD_UP)
GPIO.add_event_detect(FULL1,GPIO.RISING,MyInterrupt1,200)

#满载检测2 GPIO16引脚
def MyInterrupt2(FULL2):
    global full2flag
    global full2timeout
    time.sleep(0.5)  #Key delay times
    if GPIO.input(FULL2) == 1: #Key is low
        full2flag = 1
        full2timeout =0

GPIO.setmode(GPIO.BCM)
GPIO.setup(FULL2,GPIO.IN,GPIO.PUD_UP)
GPIO.add_event_detect(FULL2,GPIO.RISING,MyInterrupt2,200)

#满载检测3 GPIO24引脚
def MyInterrupt3(FULL3):
    global full3flag
    global full3timeout
    time.sleep(0.5)  #Key delay times
    if GPIO.input(FULL3) == 1: #Key is low
        full3flag = 1
        full3timeout =0


GPIO.setmode(GPIO.BCM)
GPIO.setup(FULL3,GPIO.IN,GPIO.PUD_UP)
GPIO.add_event_detect(FULL3,GPIO.RISING,MyInterrupt3,200)

#满载检测4 GPIO23引脚
def MyInterrupt4(FULL4):
    global full4flag
    global full4timeout
    time.sleep(0.5)  #Key delay times
    if GPIO.input(FULL4) == 1: #Key is low
        full4flag = 1
        full4timeout =0
  
        
GPIO.setmode(GPIO.BCM)
GPIO.setup(FULL4,GPIO.IN,GPIO.PUD_UP)
GPIO.add_event_detect(FULL4,GPIO.RISING,MyInterrupt4,200)

#屏幕显示线程        
def Display():
    global HazardousWaste   #有害垃圾
    global OtherWaste       #其他垃圾
    global RecyclableWaste  #可回收垃圾
    global KitchenWaste     #厨余垃圾    
    
    while True:
        if(full1flag == 1):
            ser.write(b'click b1,1')  # 屏幕组件控制指令    其他垃圾满载检测
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            print("满载检测1")

        if(full2flag == 1):
            ser.write(b'click b2,1')  # 屏幕组件控制指令    其他垃圾满载检测
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            print("满载检测2")
            
        if(full3flag == 1):
            ser.write(b'click b3,1')  # 屏幕组件控制指令    其他垃圾满载检测
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            print("满载检测3")
            
        if(full4flag == 1):
            ser.write(b'click b4,1')  # 屏幕组件控制指令    其他垃圾满载检测
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            print("满载检测4")
            
        ser.write(b't11.txt=')  #屏幕组件控制指令   有害垃圾件数
        ser.write(b'\"%d\"'%HazardousWaste) #件数   i自加 
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't16.txt=')  #屏幕组件控制指令   可回收垃圾件数
        ser.write(b'\"%d\"'%RecyclableWaste) #件数   i自加 
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't20.txt=')  #屏幕组件控制指令   厨余收垃圾件数
        ser.write(b'\"%d\"'%KitchenWaste) #件数   i自加 
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't24.txt=')  #屏幕组件控制指令   其他垃圾件数
        ser.write(b'\"%d\"'%OtherWaste) #件数   i自加 
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号

        time.sleep(0.5)
        
# 识别到的垃圾，进行分条目处理，电机控制程序可以放在每一条中
def led_select(label):
    global HazardousWaste   #有害垃圾
    global OtherWaste       #其他垃圾
    global RecyclableWaste  #可回收垃圾
    global KitchenWaste     #厨余垃圾
    global Battery  # 电池
    global BrickAndTileCeramics  # 砖瓦陶瓷
    global cigarette  # 香烟
    global Cans  # 易拉罐
    global WaterBottle  # 水瓶
    global Fruits  # 水果
    global Vegetables  # 蔬菜
    global i  # 序号
    
    #print(label)
    #识别垃圾为 电池
    if label == "battery":
        #投放装置投放垃圾
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
        time.sleep(1)
        pwm.setServoPulse(0,500+(2000/270)*90)
        time.sleep(1)
        pwm.setServoPulse(1,500+(2000/180)*0) 
        time.sleep(0.5)
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
        
        i += 1
        Battery += 1
        HazardousWaste += 1
        
        ser.write(b't3.txt=')  #屏幕组件控制指令  序号   
        ser.write(b'\"%d\"'%i) #数量   i自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't5.txt=')  #屏幕组件控制指令  垃圾种类
        ser.write(b'\"\xb5\xe7\xb3\xd8\"')  #电池     
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't7.txt=')  #屏幕组件控制指令  数量    
        ser.write(b'\"%d\"'%Battery) #数量   Battery自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't9.txt=\"OK\"')  #屏幕组件控制指令  分类是否成功
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        print("有害垃圾：电池")
        #yellow_led.on()
        
        
    #识别垃圾为 砖瓦陶瓷
    if label == "stone":
        # 投放装置投放垃圾
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
        time.sleep(1)
        pwm.setServoPulse(0,500+(2000/270)*180)
        time.sleep(1)
        pwm.setServoPulse(1,500+(2000/180)*0) 
        time.sleep(0.5)
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
         
        i += 1
        BrickAndTileCeramics +=1
        OtherWaste += 1
        
        ser.write(b't3.txt=')  #屏幕组件控制指令  序号   
        ser.write(b'\"%d\"'%i) #数量   i自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't5.txt=')  #屏幕组件控制指令  垃圾种类
        ser.write(b'\"\xd7\xa9\xcd\xdf\xcc\xd5\xb4\xc9\"')  #砖瓦陶瓷     
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't7.txt=')  #屏幕组件控制指令  数量    
        ser.write(b'\"%d\"'%BrickAndTileCeramics) #数量   i自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't9.txt=\"OK\"')  #屏幕组件控制指令  分类是否成功
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        print("其他垃圾：砖瓦陶瓷")

        
    #识别垃圾为 烟头
    if label == "cigarette":
        # 投放装置投放垃圾
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
        time.sleep(1)
        pwm.setServoPulse(0,500+(2000/270)*180)
        time.sleep(1)
        pwm.setServoPulse(1,500+(2000/180)*0) 
        time.sleep(0.5)
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
        
        i += 1
        cigarette +=1
        OtherWaste += 1
        
        ser.write(b't3.txt=')  #屏幕组件控制指令  序号   
        ser.write(b'\"%d\"'%i) #数量   i自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't5.txt=')  #屏幕组件控制指令  垃圾种类
        ser.write(b'\"\xd1\xcc\xcd\xb7\"')  #烟头  
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't7.txt=')  #屏幕组件控制指令  数量    
        ser.write(b'\"%d\"'%cigarette) #数量   i自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't9.txt=\"OK\"')  #屏幕组件控制指令  分类是否成功
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        print("其他垃圾：烟头")
        
        
    #识别垃圾为 易拉罐
    if label == "can":
        # 投放装置投放垃圾
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
        time.sleep(1)
        pwm.setServoPulse(0,500+(2000/270)*0)
        time.sleep(1)
        pwm.setServoPulse(1,500+(2000/180)*0) 
        time.sleep(0.5)
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
        
        i += 1
        Cans +=1
        RecyclableWaste += 1
        
        ser.write(b't3.txt=')  #屏幕组件控制指令  序号   
        ser.write(b'\"%d\"'%i) #数量   i自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't5.txt=')  #屏幕组件控制指令  垃圾种类    
        ser.write(b'\"\xd2\xd7\xc0\xad\xb9\xde\"')  #易拉罐     
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't7.txt=')  #屏幕组件控制指令  数量    
        ser.write(b'\"%d\"'%Cans) #数量   i自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't9.txt=\"OK\"')  #屏幕组件控制指令  分类是否成功
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        print("可回收垃圾：易拉罐")
        
        
    #识别垃圾为 矿泉水瓶
    if label == "bottle":
        # 投放装置投放垃圾
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
        time.sleep(1)
        pwm.setServoPulse(0,500+(2000/270)*0)
        time.sleep(1)
        pwm.setServoPulse(1,500+(2000/180)*0) 
        time.sleep(0.5)
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
        
        i += 1
        WaterBottle +=1
        RecyclableWaste += 1
        
        ser.write(b't3.txt=')  #屏幕组件控制指令  序号   
        ser.write(b'\"%d\"'%i) #数量   i自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't5.txt=')  #屏幕组件控制指令  垃圾种类
        ser.write(b'\"\xcb\xae\xc6\xbf\"')  #水瓶     
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't7.txt=')  #屏幕组件控制指令  数量    
        ser.write(b'\"%d\"'%WaterBottle) #水瓶数量   WaterBottle自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't9.txt=\"OK\"')  #屏幕组件控制指令  分类是否成功
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        print("可回收垃圾：矿泉水瓶")

        
    #识别垃圾为 水果
    if label == "fruit": 
        # 投放装置投放垃圾
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
        time.sleep(1)
        pwm.setServoPulse(0,500+(2000/270)*270)
        time.sleep(1)
        pwm.setServoPulse(1,500+(2000/180)*0) 
        time.sleep(0.5)
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
                
        i += 1
        Fruits +=1
        KitchenWaste += 1        
        
        ser.write(b't3.txt=')  #屏幕组件控制指令  序号   
        ser.write(b'\"%d\"'%i) #数量   i自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't5.txt=')  #屏幕组件控制指令  垃圾种类    
        ser.write(b'\"\xcb\xae\xb9\xfb\"')  #水果     
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't7.txt=')  #屏幕组件控制指令  数量    
        ser.write(b'\"%d\"'%Fruits) #数量   i自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't9.txt=\"OK\"')  #屏幕组件控制指令  分类是否成功
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
            
        print("厨余垃圾：水果")

    #识别垃圾为 蔬菜
    if label == "vegetable":
        # 投放装置投放垃圾
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*100)
        time.sleep(1)
        pwm.setServoPulse(0,500+(2000/270)*270)
        time.sleep(1)
        pwm.setServoPulse(1,500+(2000/180)*0) 
        time.sleep(0.5)
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
                
        i += 1
        Vegetables +=1
        KitchenWaste += 1
        
        ser.write(b't3.txt=')  #屏幕组件控制指令  序号   
        ser.write(b'\"%d\"'%i) #数量   i自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't5.txt=')  #屏幕组件控制指令  垃圾种类    
        ser.write(b'\"\xca\xdf\xb2\xcb\"')  #蔬菜     
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't7.txt=')  #屏幕组件控制指令  数量    
        ser.write(b'\"%d\"'%Vegetables) #蔬菜数量   i自加   
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        ser.write(b't9.txt=\"OK\"')  #屏幕组件控制指令  分类是否成功
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        ser.write(b'\xff')  #结束符号
        
        print("厨余垃圾：蔬菜")

              
    #没有垃圾
    if label == "NO_RUBBISH":
        # 投放装置等待投放垃圾
        pwm.setServoPulse(0,500+(2000/270)*135) 
        pwm.setServoPulse(1,500+(2000/180)*90)
        print("没有垃圾")     

#开启屏幕显示线程，刷新屏幕显示内容
thread = threading.Thread(target = Display, args = ())
thread.start()

# 主函数
while True:
         
    pwm = PCA9685(0x40, debug=False)
    pwm.setPWMFreq(50)
    
    if GPIO.input(KEY) == 0:        
        VideoFlag = 0 #检测到垃圾投入传感器被遮挡，标志位置0
   
    #显示屏界面切换，15S不投入垃圾，则开始播放宣传片    
    if (VideoFlag == 0): #Key is low               
        timeout += 1
        if(timeout == 15):
            ser.write(b'page video')  # 屏幕组件控制指令，宣传片界面
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
    
    if GPIO.input(FULL1) == 0: #Key is low
        full1timeout += 1
        if(full1timeout == 2):
            full1flag = 0
            ser.write(b'click b1,0')  # 屏幕组件控制指令    其他垃圾满载检测
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
        
    if GPIO.input(FULL2) == 0: #Key is low
        full2timeout += 1
        if(full2timeout == 2):
            full2flag = 0
            ser.write(b'click b2,0')  # 屏幕组件控制指令    其他垃圾满载检测
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
    
    if GPIO.input(FULL3) == 0:
        full3timeout += 1
        if(full3timeout == 2):
            full3flag = 0
            ser.write(b'click b3,0')  # 屏幕组件控制指令    其他垃圾满载检测
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
    
    if GPIO.input(FULL4) == 0:
        full4timeout += 1
        if(full4timeout == 2):
            full4flag = 0
            ser.write(b'click b4,0')  # 屏幕组件控制指令    其他垃圾满载检测
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
            ser.write(b'\xff')  # 结束符号
        
    sleep(1)#延时函数，关系到10S不投入垃圾进入宣传片播放，不可更改！
# -*- coding: utf-8 -*