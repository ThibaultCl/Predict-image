#creer un fichier Json avec les image préférées de l'utilisateur
import json
from PIL import Image
import matplotlib.pyplot as plot
import numpy
import math
from sklearn.cluster import MiniBatchKMeans

#affichage des premières images
N=8 #nombre d'image total pour le training
Ntraining=4; #nombre de choix de l'utilisateur
data=[]
data.append(["nameFile","sizeImage","listSortedMainColors","listTags"])

for j in range(0,Ntraining):
    f, axes = plot.subplots(nrows=int(N/Ntraining), ncols=1,figsize=(20,20))
    for i in range (1,int(N/Ntraining+1)):
        file='images/'+str(int(i+j*N/Ntraining))+'.jpeg'
        imgfile = Image.open(file)
        axes[i-1].imshow(imgfile)
        axes[i-1].set_title('Image '+str(int(i+j*N/Ntraining)))  
    plot.show()  

    #choix de l'image et ajout de tag par l'utilisateur
    numberImg=input("Entrer le numero de votre image préférée : ")
    tags=input("Entrer une liste de tags : ")
    tagsList=tags.split()

    #sauvegarde dans un fichier json des données de l'image
    file='images/'+numberImg+'.jpeg'
    imgfile = Image.open(file)
    numarray = numpy.array(imgfile.getdata(), numpy.uint8)
   #récupération des couleurs prédominantes
    numberCluster=6 #number of colors
    clusters = MiniBatchKMeans(n_clusters = numberCluster)
    clusters.fit(numarray)
    npbins = numpy.arange(0, numberCluster+1)
    histogram = numpy.histogram(clusters.labels_, bins=npbins)
    histogram2=sorted(histogram[0],reverse=True)

    y = []
    y.append(file)
    y.append(imgfile.size)
    x=[]
    for i in range(0,numberCluster):
        x.append([math.ceil(clusters.cluster_centers_[i][0]),math.ceil(clusters.cluster_centers_[i][1]), math.ceil(clusters.cluster_centers_[i][2])])
    y.append(x)
    y.append(tagsList)
    data.append(y)
    
with open('Image.json', 'w') as f:  # save data in Image.json
    json.dump(data, f)
print("\nSauvegarde des données dans Image.json")
print (data)
