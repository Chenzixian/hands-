import cv2
import mediapipe as mp
import  time

cap = cv2.VideoCapture(0) #摄像头权限

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.7) #告诉计算机我要做手部跟踪项目
mpDraw = mp.solutions.drawing_utils

handLmsStyle =  mpDraw.DrawingSpec(color=(0,0,255),thickness=2)  #设定点(handLmsStyle)的颜色(color)、粗细(thickness)
handConStyle= mpDraw.DrawingSpec(color=(0,255,0),thickness=1)  #设定线（handConStyle)的颜色(color)、粗细(thickness)

pTime = 0
cTime = 0

while True:
    ret, img = cap.read()

    if ret :
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        #print(result.multi_hand_landmarks)

        imgHeight = img.shape[0] #设置视窗高
        imgWidth = img.shape[1] #设置视窗宽

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks: #把21个点和线画出来
                mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS,handLmsStyle,handConStyle) #第四个设定点的样式，第五个参数设置线的样式
                for i,lm in enumerate(handLms.landmark): #i=第几个点，lm=点的坐标
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)

                    #cv2.putText(img,str(i),(xPos-25,yPos+5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2)#显示24个点的数字
                    #放大某个点
                    #if i ==4:
                        #cv2.circle(img,(xPos,yPos),20,(166,56,56),cv2.FILLED)

                    print(i,xPos,yPos)
        #显示帧数
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,f"FPS:{int(fps)}", (30,50),  cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

        cv2.imshow('img',img)

        if cv2.waitKey(1) == ord('q'):
            break