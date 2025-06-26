import serial
import time
from time import sleep
import numpy as np
ser = serial.Serial("/dev/ttyAMA1", 9600)
mesh_min=[1,1]#网格左上角坐标
mesh_max=[12,13]#网格右下角坐标
pix_min=[564,152]#网格左上角像素
pix_max=[1204,729]#网格右下角像素
pos=mesh_min#代表[行,列]
port=[mesh_min,[mesh_max[0],mesh_min[1]],mesh_max,[mesh_min[0],mesh_max[1]]]#逆时针
allpulse=[105000,22000]
step=[allpulse[0]//(mesh_max[0]-mesh_min[0]),allpulse[1]//(mesh_max[1]-mesh_min[1])]#走一步需要的脉冲
moving=[False,False,False,False]
pix_mesh_conv=[(pix_max[0]-pix_min[0])//(mesh_max[0]-mesh_min[0]),(pix_max[1]-pix_min[1])//(mesh_max[1]-mesh_min[1])]
#进是远离舵机位置
direction=[0,1,0,0]
servo=[[0x01,0xfd,0x12,0xbc,0xfe,0x01,0x9a,0x28,0x6b],#1:0进1退
       [0x02,0xfd,0x01,0x2c,0xfe,0x00,0x55,0xf0,0x6b],#2:0退1进
       [0x03,0xfd,0x13,0x32,0xfe,0x00,0x0f,0xa0,0x6b],#3:0进1退，第三位13或00
       [0x04,0xfd,0x11,0xf4,0xfe,0x00,0x27,0x10,0x6b],#4:1收
       [0x04,0xfd,0x01,0xf4,0xfe,0x00,0x27,0x10,0x6b]]#4:0开
up=[0x03,0xfd,0x12,0x58,0xfe,0x00,0x03,0xe8,0x6b]
donestr=["019f6b","029f6b","039f6b","049f6b"]
def go_port(n):
    des=port[n]
    global pos,moving
    for i in range(2):
        if pos[i]!=des[i]:
            moving[i]=True
            move=servo[i]
            pulse=abs(pos[i]-des[i])*step[i]
            if pos[i]>des[i]:
                move[2]=move[2]%16+direction[i]*16
            else:
                move[2]=move[2]%16+(1-direction[i])*16
            print(pulse)
            move[7]=pulse%256
            move[6]=pulse//256%256
            move[5]=pulse//256//256%256
            print(move)
            ser.write(move)
            time.sleep(0.02)
    t1=time.time()
    while 1:
        read=ser.read_all().hex()
        for i in range(4):
            if moving[i]:
                if donestr[i] in read:
                    moving[i]=False
        if (moving[0] or moving[1] or moving[2] or moving[3])==False:
            print("done")
            break
        t2=time.time()
        if t2-t1>3:
            moving=[False,False,False,False]
            print("timeout")
            break      
    pos=des
    time.sleep(1)
    drop()
    print(pos)
def go_pos(pix):
    if pix[0]>=pix_min[0] and pix[1]>=pix_min[1] and pix[0]<=pix_max[0] and pix[1]<=pix_max[1]:
        des=[(pix[0]-pix_min[0])//pix_mesh_conv[0]+1,(pix[1]-pix_min[1])//pix_mesh_conv[1]+1]
        global pos,moving
        for i in range(2):
            if pos[i]!=des[i]:
                moving[i]=True
                move=servo[i]
                pulse=abs(pos[i]-des[i])*step[i]
                if pos[i]>des[i]:
                    move[2]=move[2]%16+direction[i]*16
                else:
                    move[2]=move[2]%16+(1-direction[i])*16
                print(pulse)
                move[7]=pulse%256
                move[6]=pulse//256%256
                move[5]=pulse//256//256%256
                print(move)
                ser.write(move)
                time.sleep(0.02)
        t1=time.time()
        while 1:
            read=ser.read_all().hex()
            for i in range(3):
                if moving[i]:
                    if donestr[i] in read:
                        moving[i]=False
            if (moving[0] or moving[1] or moving[2] or moving[3])==False:
                print("done")
                break
            t2=time.time()
            if t2-t1>3:
                moving=[False,False,False,False]
                print("timeout")
                break      
        pos=des
        time.sleep(1)
        grab()
    else:
        print("out of range")
    print(pos)
def grab():
    ser.write(servo[3])
    time.sleep(0.02)
    t1=time.time()
    while 1:
        read=ser.read_all().hex()
        if donestr[3] in read:
            print(read)
            print("grasped")
            time.sleep(1)
            break
        t2=time.time()
        if t2-t1>3:
            print("grasp timeout")
            break
def drop():
    ser.write(servo[4])
    time.sleep(0.02)
    t1=time.time()
    while 1:
        read=ser.read_all().hex()
        if donestr[3] in read:
            print(read)
            print("dropped")
            time.sleep(1)
            break
        t2=time.time()
        if t2-t1>3:
            print("drop timeout")
            break
while 1:
    go_pos([800,500])#走到位置，合上爪子
    go_port(0)#走到垃圾桶，张开爪子
    