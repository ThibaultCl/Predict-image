#utilisation d'un reseau de neuronne pour prédire les préférences de l'utilisateur sur une image
import json
import os,sys
from PIL import Image
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plot
import math
from sklearn import datasets, svm, metrics


#lecture du fichier json
data=""
with open("Image.json", "r") as file:
    data=json.load(file)
    
print("lecture du fichier user :")
print(data)
print("\n")

#déclaré dans la céllule précédente
#N=8
#Ntraining=2

#récupéation des images
images=[]
for i in range(1,N+1):
    file="images/"+str(i)+".jpeg"
    imgfile = Image.open(file)
    imgfile = imgfile.resize([300,300], Image.ANTIALIAS)
    images.append(list(map(list, list(imgfile.getdata()))))

#reshape de la liste de chaque image[i]
for i in range(0,len(images)):
    images[i]=numpy.array(images[i])
    images[i]=images[i].reshape(1,-1).flatten().tolist()

training_images=numpy.array(images)

#création de la liste de label 
label=[]
for i in range(0,N):
    label.append(0)

for j in range(1,len(data)):
    label[int(data[j][0][7:-5])-1]=1

training_target=numpy.array(label)

classifier = Perceptron(max_iter=1000)

#training
classifier.fit(training_images, training_target)

print("Apprentissage terminé, ",N, " images renseignées")    


#prediction
real_label=[]
predicted_labels=[]
number_images=[]
j=0
print("Suggestion d'images que vous pourriez aimer :")

#parcours les images de test
for i in range(N+1,18):
    #récupération de l'image
    file="images/"+str(i)+".jpeg"
    imgfile=Image.open(file)
    predict_image =  numpy.array(imgfile.getdata())
    predict_image =  predict_image.reshape(1,-1)
    
    #prédiction du label
    predicted_labels= numpy.append(predicted_labels,classifier.predict(predict_image)[0])
    
    #si le label prédit =1 alors l'image doit plaire à l'utilisateur et on lui demande
    if(predicted_labels[j]==1):
        plot.imshow(imgfile)
        plot.show()
        real_label=numpy.append(real_label, int(input("Cette image vous plait-elle ? 0=non 1=oui :")))
    else:
         real_label=numpy.append(real_label,0)
    j+=1
    number_images.append(i)

print('\n')
print("numero des images",number_images)
print("predicted_labels",predicted_labels)
print("real_label",real_label)
print('\n')
print(metrics.classification_report(real_label,predicted_labels))



