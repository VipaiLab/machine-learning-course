import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

input_filename = '001.jpg'
output_filename = '001Detected.jpg'
img1 = cv2.imread(input_filename)
img = cv2.resize(img1,(240,320),interpolation=cv2.INTER_LINEAR)

faces = face_cascade.detectMultiScale(img,1.2,2)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#用颜色为BGR（255,0,0）粗度为2的线条在img画出识别出的矩型
    face_re = img[y:y+h,x:x+w]#抽取出框出的脸部部分，注意顺序y在前
    eyes = eye_cascade.detectMultiScale(face_re)#在框出的脸部部分识别眼睛
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(face_re,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imwrite(output_filename,img)
