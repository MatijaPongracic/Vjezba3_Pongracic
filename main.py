import cv2
import os

face1 = "opencv_face_detector.pbtxt"
face2 = "opencv_face_detector_uint8.pb"
age1 = "age_deploy.prototxt"
age2 = "age_net.caffemodel"
gen1 = "gender_deploy.prototxt"
gen2 = "gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603,87.7689143744,114.895847746)

face = cv2.dnn.readNet(face2,face1)
age = cv2.dnn.readNet(age2,age1)
gen = cv2.dnn.readNet(gen2,gen1)

la = ["(0-2)","(4-6)","(8-12)","(15-20)","(25-32)","(38-43)","(48-53)","(60-100)"]
lg = ["Male","Female"]

def slika(img):
    image = cv2.imread(img)
    fr_cv = image.copy()
    fr_h,fr_w,_ = fr_cv.shape
    blob = cv2.dnn.blobFromImage(fr_cv, 1.0, (300,300), [104,117,123],True,False)
    face.setInput(blob)
    detections = face.forward()

    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence>0.7:
            x1 = int(detections[0,0,i,3]*fr_w)
            y1 = int(detections[0,0,i,4]*fr_h)
            x2 = int(detections[0,0,i,5]*fr_w)
            y2 = int(detections[0,0,i,6]*fr_h)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(fr_cv, (x1,y1), (x2,y2), (0,255,0),4,8)

    for faceBox in faceBoxes:
        detected_face = fr_cv[max(0, faceBox[1]-15) : min(faceBox[3]+15, fr_cv.shape[0]-1),
                              max(0, faceBox[0]-15) : min(faceBox[2]+ 5, fr_cv.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(detected_face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB = False)

        gen.setInput(blob)
        genderPreds = gen.forward()
        spol = lg[genderPreds[0].argmax()]

        age.setInput(blob)
        agePreds = age.forward()
        dob = la[agePreds[0].argmax()]

        cv2.putText(fr_cv,
                    f"{spol}, {dob}",
                    (faceBox[0],faceBox[1]+10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,255),
                    2,
                    cv2.LINE_AA)

    project_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(project_path, "Slika_output.jpg")
    cv2.imwrite(output_path, fr_cv)

    cv2.imshow(img,fr_cv)
    cv2.waitKey(0)

def video():
    cap = cv2.VideoCapture(0)

    while True:
        ret,frame = cap.read()
        if not ret:
            print("Kamera ne radi!")
            break

        fr_h,fr_w,_ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], True,False)
        face.setInput(blob)
        detections = face.forward()

        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence>0.7:
                x1 = int(detections[0,0,i,3]*fr_w)
                y1 = int(detections[0,0,i,4]*fr_h)
                x2 = int(detections[0,0,i,5]*fr_w)
                y2 = int(detections[0,0,i,6]*fr_h)
                faceBoxes.append([x1,y1,x2,y2])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 4, 8)

        for faceBox in faceBoxes:
            detected_face = frame[max(0, faceBox[1]-15) : min(faceBox[3]+15, frame.shape[0]-1),
                                  max(0, faceBox[0]-15) : min(faceBox[2]+5, frame.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(detected_face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB = False)

            gen.setInput(blob)
            genderPreds = gen.forward()
            spol = lg[genderPreds[0].argmax()]

            age.setInput(blob)
            agePreds = age.forward()
            dob = la[agePreds[0].argmax()]

            cv2.putText(frame,
                        f'{spol}, {dob}',
                        (faceBox[0],faceBox[1]+10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,255),
                        2,
                        cv2.LINE_AA)

        cv2.imshow("Kamera",frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

slika("Proba.jpg")
video()