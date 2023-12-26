import cv2
import os
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

class Language():
    def __init__(self,welcome,odabir,slika,kamera,spol,dob,male,female):
        self.welcome = welcome
        self.odabir = odabir
        self.slika = slika
        self.kamera = kamera
        self.spol = spol
        self.dob = dob
        self.male = male
        self.female = female

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

def resize(image_path, new_width, new_height):
    original_image = Image.open(image_path)
    resized_image = original_image.resize((new_width,new_height))
    return ImageTk.PhotoImage(resized_image)

def slika(img,lang,buttonPho,buttonCam):
    buttonPho.configure(state = DISABLED)
    buttonCam.configure(state = DISABLED)
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
        if spol == "Male":
            spol = lang.male
        else:
            spol = lang.female

        age.setInput(blob)
        agePreds = age.forward()
        dob = la[agePreds[0].argmax()]

        cv2.putText(fr_cv,
                    f"{lang.spol}: {spol}",
                    (faceBox[0], faceBox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA)

        cv2.putText(fr_cv,
                    f"{lang.dob}: {dob}",
                    (faceBox[0], faceBox[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA)

    project_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(project_path, "Slika_output.jpg")
    cv2.imwrite(output_path, fr_cv)

    cv2.imshow(img,fr_cv)
    cv2.waitKey(0)
    buttonPho.configure(state=ACTIVE)
    buttonCam.configure(state=ACTIVE)

def video(lang,buttonPho,buttonCam):
    buttonPho.configure(state=DISABLED)
    buttonCam.configure(state=DISABLED)

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
            if spol == "Male":
                spol = lang.male
            else:
                spol = lang.female

            age.setInput(blob)
            agePreds = age.forward()
            dob = la[agePreds[0].argmax()]

            cv2.putText(frame,
                        f"{lang.spol}: {spol}",
                        (faceBox[0],faceBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,255),
                        2,
                        cv2.LINE_AA)

            cv2.putText(frame,
                        f"{lang.dob}: {dob}",
                        (faceBox[0], faceBox[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA)

        cv2.imshow("Kamera",frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    buttonPho.configure(state=ACTIVE)
    buttonCam.configure(state=ACTIVE)

def chooseFile(lang,buttonPho,buttonCam):
    filepath = filedialog.askopenfilename(filetypes = [("Photos","*.png;*.jpg;*.jpeg")])
    if filepath:
        slika(filepath,lang,buttonPho,buttonCam)

def choose(window, lang):
    for widget in window.winfo_children():
        widget.destroy()

    label1 = Label(text = f"{lang.welcome}\n{lang.odabir}",
                  font = ("Arial", 40, "bold"),
                  fg = "white",
                  bg = "black",
                  pady = 50)
    label1.pack()

    buttonPho = Button(text=lang.slika,
                       font=("Arial", 20),
                       fg="blue",
                       bg="black",
                       command=lambda: chooseFile(lang,buttonPho,buttonCam),
                       activeforeground="blue",
                       activebackground="black",
                       image=photoPho,
                       compound="top")
    buttonPho.pack()

    buttonCam = Button(text=lang.kamera,
                      font=("Arial", 20),
                      fg="blue",
                      bg="black",
                      command=lambda: video(lang,buttonPho,buttonCam),
                      activeforeground="blue",
                      activebackground="black",
                      image=photoCam,
                      compound="top")
    buttonCam.pack()

window = Tk()
window.geometry("840x630")
window.title("Unaprijeđena zadaća")
window.config(background = "black")
photoCro = resize("Hrvatska.png",120,60)
photoUK = resize("UK.png",120,60)
photoSwe = resize("Sverige.png",120,60)
photoPho = resize("Fotoaparat.png",100,100)
photoCam = resize("Kamera.png",100,100)
hrvatski = Language("Dobrodošli!","Što želite odabrati?","Slika","Kamera","Spol","Dob","Musko","Zensko")
english = Language("Welcome!","What do you want to choose?","Photo","Camera","Sex","Age","Male","Female")
svenska = Language("Välkommen!","Vad vill du välja?","Foto","Kamera","Sex","Alder","Man","Kvinna")

label = Label(text = "Choose your language:",
              font = ("Arial",40,"bold"),
              fg = "white",
              bg = "black",
              pady = 50)
label.pack()

buttonCro = Button(text = "Hrvatski",
                   font = ("Arial",20),
                   fg = "blue",
                   bg = "black",
                   command = lambda: choose(window, hrvatski),
                   activeforeground = "blue",
                   activebackground = "black",
                   image = photoCro,
                   compound = "top")
buttonCro.pack()

buttonUK = Button(text = "English",
                   font = ("Arial",20),
                   fg = "blue",
                   bg = "black",
                   command = lambda: choose(window, english),
                   activeforeground = "blue",
                   activebackground = "black",
                   image = photoUK,
                   compound = "top")
buttonUK.pack()

buttonSwe = Button(text = "Svenska",
                   font = ("Arial",20),
                   fg = "blue",
                   bg = "black",
                   command = lambda: choose(window, svenska),
                   activeforeground = "blue",
                   activebackground = "black",
                   image = photoSwe,
                   compound = "top")
buttonSwe.pack()

window.mainloop()