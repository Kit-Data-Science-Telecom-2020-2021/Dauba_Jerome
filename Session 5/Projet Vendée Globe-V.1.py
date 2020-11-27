#!/usr/bin/env python
# coding: utf-8

#  # <center><u> Projet final du Kit Data Science 2020

#  <center> Le Vendée Globe est à ce jour la plus grande course à la voile autour du monde, en solitaire, sans escale et sans assistance. L'événement s'est inscrit dans le sillage du Golden Globe qui, en 1968, initia la première circum navigation de ce type par les trois caps (Bonne Espérance, Leeuwin et Horn). Sur les neuf pionniers à s'élancer en 1968 un seul réussit à revenir à Falmouth, le grand port de la Cornouailles anglaise. Le 6 avril 1969 après 313 jours de mer, le Britannique Robin Knox-Johnston arrivait enfin au but. Vingt années plus tard, c'est le navigateur Philippe Jeantot qui, après sa double victoire dans le BOC Challenge (Le tour du monde en solitaire avec escales), lance l'idée d'une nouvelle course autour du monde, en solitaire, mais... sans escale ! Le Vendée Globe était né. Le 26 novembre 1989, treize marins prennent le départ de la première édition qui durera plus de trois mois. Ils ne seront que sept à rentrer aux Sables d'Olonne.

#  <center>Les huit éditions de ce que le grand public nomme aujourd'hui l'Everest des mers, ont permis à 167 concurrents de prendre le départ de cette course hors du commun. Seuls 89 d'entre eux ont réussi à couper la ligne d'arrivée. Ce chiffre exprime à lui seul l'extrême difficulté de cet événement planétaire où les solitaires sont confrontés au froid glacial, aux vagues démesurées et aux ciels pesants qui balayent le grand sud ! Le Vendée Globe est avant tout un voyage au bout de la mer et aux tréfonds de soi-même. Le neuvième Vendée Globe s'est élancé des Sables d'Olonne le dimanche 8 novembre 2020.

# <img src="https://www.safetics.com/wp-content/uploads/2018/06/Logo_Safetics_Entreprise_Vend%C3%A9e-Globe.jpg">
# + Montage photo avec bateau + télécom sur Github ?

# #### Dans ce rapport, nous utiliserons Pandas et le cours de "Kit Data Science" afin d'acquérir les données **à jour** du Vendée Globe 2020, les traiter, pour ensuite apporter une analyse pertinente sur celles-ci.
# 
# #### <u>Nous repésenterons dans un premier temps ces données de manière claire... :
# - Carte
# - Fiche decription (pivot table)
# - Classement à récupérer tous les jours !
# 
# #### <u>... Avant de mettre en relation ces données pour faire ressortir des analyses pertinentes :
# - Corélation entre nombre de foils et vitesse moyenne
# - etc...

# #### Sources :
# - https://www.vendeeglobe.org/fr/classement
# - https://www.vendeeglobe.org/fr/glossaire
# - ...

# #### <u> Fonctionnement du Jupyter Notebook:   
# L'ensemble de toutes les fonctions sont écrite en début de chaque partie du rapport. Elles sont ensuite appelées au cours du Notebook.
# 
# Pour mettre à jour les données, il est **nécessaire de Run les cellules avant lecture du Notebook !**

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import re
import random
import warnings 
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
import requests
from bs4 import BeautifulSoup
import datetime
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
import os
try:
    from mpl_toolkits.basemap import Basemap
except:
    os.environ["PROJ_LIB"] = "C:\\Users\\jerom\\Anaconda3\\Library\\share"; # Petit fix perso 
    from mpl_toolkits.basemap import Basemap


# In[3]:


# On définit ici quelques fonctions d'aide :

# Vérifier si un Skipper est dans le Top 3:
def isintop3(x):
    if x <=3:
        return(1)
    else :
        return(0)

# Transformation Longitude / Latitude :
def get_latitude(x):
    minut = int(int(x[3:5])*100/60)
    if x[-1]=="S":
        lat = x[:2]+"."+f"{minut}"
        lat = -float(lat)
    elif x[-1]=="N":
        lat = x[:2]+"."+f"{minut}"
        lat = float(lat)
    return(lat)

def get_longitude(x):
    minut = int(int(x[3:5])*100/60)
    if x[-1]=="W":
        lon = x[:2]+"."+f"{minut}"
        lon = -float(lon)
    elif x[-1]=="E":
        lon = x[:2]+"."+f"{minut}"
        lon = float(lon)
    return(lon)


# Date du jour :
date = datetime.datetime.now()
heure = date.hour
if heure > 16 : 
    today = date.day
else :
    today = date.day -1

# Liste avec les dates jusqu'aujourd'hui :
liste_dates = []
for i in range(8,today+1):
    if i < 10 :
        i = '0'+f'{i}'
    liste_dates.append(f"11/{i}/2020")

# Fonction pour détecter la présence d'un foil dans le DataFrame 'fiches':
def presence_foils(x):
    if x == "foils":
        return(1)
    else :
        return(0)


# In[ ]:


# On définit ici les fonctions essentielles pour la partie 1, concernant les classements, fiches techniques
#  et fusion des données

# Importation et traitement de l'Excel principal du Vendée Globe (A mettre dans une fonction après !)
# On va chercher tous les classements journaliers (1 fois / jour) jusqu'aujourd'hui :

def classement():
    i=0
    for day in range(8,today+1):
        if day < 10:
            day = '0'+f"{day}"
        url_type = "https://www.vendeeglobe.org/download-race-data/vendeeglobe_202011"+f"{day}_140000.xlsx"
        df = pd.read_excel(url_type, skiprows=range(1,5),nrows=33,usecols=range(3,21),keep_default_na=False,names=["Skipper/Bateau", "Heure FR", "Latitude", "Longitude", "Cap", "Vitesse", "VMG", "Distance", "Cap_dernier_classement", "Vitesse_dernier_classement", "VMG_dernier_classement", "Distance_dernier_classement", "Cap_journalier", "Vitesse_journalier", "VMG_journalier", "Distance_journalier", "DTF", "DTL"])
        df.insert(0, "Date",f"11/{day}/2020")

        # On sépare les données dans de nouvelles colonnes pour une meilleure visibilité :
        df.insert(1,"Skipper", df["Skipper/Bateau"].str.split("\n").apply(lambda x: x[0]))
        df.insert(2,"Bateau", df["Skipper/Bateau"].str.split("\n").apply(lambda x: x[1]))
        df.drop(columns = "Skipper/Bateau", inplace=True)

        # On nettoie les colonnes en enlevant les caractères inutiles (ex : unités)
        df["Heure FR"] = df["Heure FR"].str[:5]
        df["Bateau"] = df["Bateau"].apply(lambda x: x.upper())
        df["Cap"], df["Cap_dernier_classement"], df["Cap_journalier"] = df["Cap"].str.extract(r'(\d+)'), df["Cap_dernier_classement"].str.extract(r'(\d+)'), df["Cap_journalier"].str.extract(r'(\d+)')
        df["Vitesse"], df["Vitesse_dernier_classement"], df["Vitesse_journalier"] = df["Vitesse"].str.extract(r'(\d+\.\d)'), df["Vitesse_dernier_classement"].str.extract(r'(\d+\.\d)'), df["Vitesse_journalier"].str.extract(r'(\d+\.\d)')
        df["VMG"], df["VMG_dernier_classement"], df["VMG_journalier"] = df["VMG"].str.extract(r'(\d+\.\d)'), df["VMG_dernier_classement"].str.extract(r'(\d+\.\d)'), df["VMG_journalier"].str.extract(r'(\d+\.\d)')
        df["Distance"], df["Distance_dernier_classement"], df["Distance_journalier"] = df["Distance"].str.extract(r'(\d+\.\d)'), df["Distance_dernier_classement"].str.extract(r'(\d+\.\d)'), df["Distance_journalier"].str.extract(r'(\d+\.\d)')
        df["DTF"], df["DTL"] = df["DTF"].str.extract(r'(\d+\.\d)'), df["DTL"].str.extract(r'(\d+\.\d)')

        # On convertit les strings en float pour les utiliser par la suite :
        convert_dict = {"Cap":float, "Vitesse":float, "Vitesse_dernier_classement":float, "Vitesse_journalier":float, "VMG":float, "Distance":float,"Cap_dernier_classement":float,"VMG_dernier_classement":float, "Distance_dernier_classement":float, "Cap_journalier":float, "VMG_journalier":float,"Distance_journalier":float,"DTF":float,"DTL":float}
        df = df.astype(convert_dict)

        if i == 0:
            df_classement = df
        else :
            df_classement = pd.concat([df_classement,df])
        i+=1

    df_classement.insert(0,"Classement", df_classement.index+1)
    df_classement["Date"]= pd.to_datetime(df_classement["Date"])
    df_classement["DATE"]=df_classement["Date"]
    df_classement = df_classement.set_index("Date")
    df_classement.sort_values(["Date","Classement"], ascending=[False, True], inplace=True)
    return(df_classement)

df_classement = classement()

# Extraction des fiches techniques pour les voiliers : sous forme de DataFrame puis de Pivot Table ?
def fiches():
    url_fiches = "https://www.vendeeglobe.org/fr/glossaire"

    # def extract_fiches():
    response = requests.get(url_fiches)
    soup = BeautifulSoup(response.content)
    info_bateaux = soup.find_all('li')[77:526] # On extrait que les infos techniques utiles
    info_bateau = soup.find_all("h3") # On extrait que les noms des bateaux

    # On nettoie les données en enlevant les balises :
    info_clean = []
    info_bateau_clean = []
    for i in info_bateaux:
        info_clean.append(i.get_text())
    for i in info_bateau:
        info_bateau_clean.append(i.get_text())

    # On construit une liste de dictionnaires contenant les infos de chaque bateau :
    output = []
    tmp = {}

    elem_old = "Numéro de voile"
    for elem in info_clean:
        elem = elem.split(" : ")
        if (elem[0]=="Numéro de voile") and len(tmp)!=0 :
            output.append(tmp)
            tmp = {}
        if elem[0]=="Architecte" and elem[1]=="Verdier" and elem_old!="Numéro de voile" : # Car sur la page web, 1 skipper n'a pas de N° de voile
            output.append(tmp)
            tmp = {}
        tmp[elem[0]]=elem[1]
        elem_old = elem[0]
    output.append(tmp)

    # On récupère le nom des colonnes de notre futur Dataframe pour les fiches bateau :
    datacolumns = set()
    for data in output:
        datacolumns = datacolumns.union(set(data.keys()))

    # On a plus qu'à créer le  DataFrame :
    df_fiches = pd.DataFrame(data=output, columns=datacolumns) #.reindex(sorted(df_fiches.columns), axis=1)
    df_fiches.insert(0, "Bateau",set(info_bateau_clean[1:-1]))
    df_fiches["Bateau"] = df_fiches["Bateau"].apply(lambda x: str.strip(x)).apply(lambda x: x.upper())

    # Map pour avoir des noms de bateaux identiques dans tous les DataFrames :
    dico_bateaux = {"NEWREST - ART & FENÊTRES":"NEWREST - ART ET FENETRES", "BUREAU VALLEE 2":"BUREAU VALLÉE 2", "COMPAGNIE DU LIT / JILITI":"COMPAGNIE DU LIT - JILITI", "CORUM L'EPARGNE":"CORUM L'ÉPARGNE", "INITIATIVES-COEUR":"INITIATIVES - COEUR", "PURE - BEST WESTERN®":"PURE - BEST WESTERN HOTELS AND RESORTS", "TSE -  4MYPLANET":"TSE - 4MYPLANET", "V AND B-MAYENNE":"V AND B MAYENNE", "YES WE CAM!":"YES WE CAM !"}
    df_fiches["Bateau"] = df_fiches["Bateau"].map(dico_bateaux).fillna(df_fiches["Bateau"])

    # On supprime les unités :
    df_fiches["Longueur"] = df_fiches["Longueur"].str.extract(r'(\d+)')
    df_fiches["Tirant d'eau"] = df_fiches["Tirant d'eau"].str.extract(r'(\d+)')
    df_fiches["Largeur"] = df_fiches["Largeur"].str.extract(r'(\d+)')
    df_fiches["Surface de voiles au portant"] = df_fiches["Surface de voiles au portant"].str.extract(r'(\d+)')
    df_fiches["Surface de voiles au près"] = df_fiches["Surface de voiles au près"].str.extract(r'(\d+)')
    df_fiches["Déplacement (poids)"] = df_fiches["Déplacement (poids)"].str.extract(r'(\d+)')
    df_fiches["Hauteur mât"] = df_fiches["Hauteur mât"].str.extract(r'(\d+)')

    # On convertit les strings en float :
    convert_dict2 = {"Longueur":float, "Tirant d'eau":float, "Largeur":float,"Surface de voiles au portant":float, "Surface de voiles au près":float, "Déplacement (poids)":float, "Hauteur mât":float}
    df_fiches = df_fiches.astype(convert_dict2)
    # On crée une colonne supplémentaire qui vérifie la présence de foils ou non:
    df_fiches.insert(8, "Presence foils",df_fiches["Nombre de dérives"].apply(presence_foils))
    return(df_fiches)

df_fiches = fiches()

# On va merge le DataFrame des fiches avec le Dataframe de classement :
def merged():
    df_merged = pd.merge(df_classement, df_fiches, right_on="Bateau", left_on="Bateau") 
    df_merged.index = df_merged["DATE"]
    return(df_merged)

df_merged = merged()


# In[4]:


# On définit ici les fonctions utiles à la représentation visuelle de la course

def draw_chartrace(day):
    n = 7
    df_chartrace = df_merged[["Skipper", "Bateau", "DTF", "Classement"]]
    df_chartrace.index = df_merged.index
    colors = ['red','brown', 'orange', 'greenyellow', 'blue', 'cyan','brown','purple', "gray", "cyan", "magenta", "tomato", "chocolate", "peru", "navy", "plum", "firebrick", "darkolivegreen", "lawngreen", "lime", "g", "red", "green", "blue", "purple", "gray", "deepskyblue", 'cyan','orange','purple', "gray", "hotpink", "darkblue"]
    dict_colors = dict(zip(df_merged["Skipper"].unique(), colors))
    df_chartrace["Distance to arrival"] = 36000 - df_chartrace["DTF"]
    df_chartrace = df_chartrace.loc[df_chartrace["Classement"]<=n]
    df_chartrace.sort_values("Classement",ascending=False, inplace=True)
    
    df = df_chartrace.loc[df_chartrace.index==day] #OK
    
    
    ax.clear()
    ax.barh(df['Skipper'], df['Distance to arrival'], color=[dict_colors[x] for x in df["Skipper"]])
    mini = df["Distance to arrival"].min()
    maxi = df["Distance to arrival"].max()
    ax.set_xlim(mini-500,maxi+100)
    dx = df['Distance to arrival'].max() / 2000
    for i, (distance, skipper) in enumerate(zip(df['Distance to arrival'], df['Skipper'])):
        ax.text(distance-dx, i,     skipper,           size=16, weight=600, ha='right', va='bottom')
        ax.text(distance+dx, i,     f'{distance:,.0f}',  size=14, ha='left',  va='center')
        ax.text(1, 0.4, day[3:5], transform=ax.transAxes, color='#777777', size=40, ha='right', weight=800)
        ax.text(0, 1.06, 'Distance à l arrivée (en Km)', transform=ax.transAxes, size=12, color='#777777')
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', colors='#777777', labelsize=12)
        ax.set_yticks([])
        ax.margins(0, 0.01)
        ax.grid(which='major', axis='x', linestyle='-')
        ax.set_axisbelow(True)
        ax.text(0, 1.15, f'Vendée Globe 2020 - Distance to arrival - Top{n}',
                transform=ax.transAxes, size=24, weight=600, ha='left', va='top')
        ax.text(1, 0, 'by Jérôme Dauba - MS IA', transform=ax.transAxes, color='#777777', ha='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
        plt.box(False)
 

############################################################################
# Fonction pour représenter la course sur une carte du monde, aujourd'hui !
#On utilise ici la projection de lambert :

def geoplot_ajd_lambert():

    day = datetime.datetime.now().day
    df_geo = df_classement[["DATE", "Skipper", "Latitude", "Longitude", "Cap"]]
    df_coord = df_geo.loc[df_geo["DATE"]==f"11/{day}/2020", ["Longitude", "Latitude"]]
    df_firstskipper = df_geo.loc[df_geo["DATE"]==f"11/{day}/2020", "Skipper"].to_numpy()[0]
    df_coord = df_coord.to_numpy()
    df_coord = df_coord[:-1,:]

    liste_coord = []
    for skipper in df_coord:
        if skipper.any() != "":
            long = get_longitude(skipper[0])
            lat = get_latitude(skipper[1])
            liste_coord.append([long, lat])

    plt.figure(figsize=(14, 14))
    m = Basemap(width=12000000,height=9000000,projection='lcc',
                resolution='l',lat_0=liste_coord[0][1],lon_0=liste_coord[0][0])
    m.bluemarble(scale=0.5);
    m.etopo(scale=0.5, alpha=0.5)
    m.drawcountries(linewidth=0.2)
    # m.drawcoastlines()
    for i in liste_coord:
        x,y = m(i[0], i[1])
        plt.plot(x, y, 'd', markersize=15)
        if i == liste_coord[0]:
            plt.text(x, y, df_firstskipper, fontsize=15);


    plt.title("Position en direct - Vendée Globe 2020", size=20)
    plt.show()

    
############################################################################
# Fonction pour représenter la course sur une carte du monde, aujourd'hui !
#On utilise ici la projection sinusoidale :

def geoplot_ajd_sinu():

    day = datetime.datetime.now().day
    df_geo = df_classement[["DATE", "Skipper", "Latitude", "Longitude", "Cap"]]
    df_coord = df_geo.loc[df_geo["DATE"]==f"11/{day}/2020", ["Longitude", "Latitude"]]
    df_firstskipper = df_geo.loc[df_geo["DATE"]==f"11/{day}/2020", "Skipper"].to_numpy()[0]
    df_coord = df_coord.to_numpy()
    df_coord = df_coord[:-1,:]

    liste_coord = []
    for skipper in df_coord:
        if skipper.any() != "":
            long = get_longitude(skipper[0])
            lat = get_latitude(skipper[1])
            liste_coord.append([long, lat])

    plt.figure(figsize=(14, 14))

    m = Basemap(width=12000000,height=9000000,projection='sinu',
                resolution='l',lat_0=liste_coord[0][1],lon_0=liste_coord[0][0])
    m.bluemarble(scale=0.5);
    m.etopo(scale=0.5, alpha=0.5)
    m.drawcountries(linewidth=0.2)
    m.drawcoastlines()

    # Double boucle pour les différentes dates ! Etape d'après : afficher les noms et assigner une couleur.
    # N'afficher que le nom du premier !
    # Plot points : m(long, lat)


    for i in liste_coord:
        x,y = m(i[0], i[1])
        plt.plot(x, y, 'd', markersize=10)
        if i == liste_coord[0]:
            plt.text(x, y, df_firstskipper, fontsize=15);


    plt.title("Position en direct - Vendée Globe 2020", size=20)
    plt.show()
    

############################################################################
# Fonction pour représenter la course sur une carte du monde, et ce depuis LE DEBUT DE LA COURSE !
# On utilise ici la projection de lambert :
###### Attention : le temps de calcul pour afficher la course dynamique prend 1min30.

def geoplot(day):
    # On crée un dataset avec toutes les positions des skippers :
    df_geo = df_classement[["DATE", "Skipper", "Latitude", "Longitude", "Cap"]]
    df_coord = df_geo.loc[df_geo["DATE"]==day, ["Longitude", "Latitude"]]
    df_firstskipper = df_geo.loc[df_geo["DATE"]==day, "Skipper"].to_numpy()[0]
    df_coord = df_coord.to_numpy()
    df_coord = df_coord[:-1,:]
    
    # On transforme les positions GPS pour qu'elles soient utilisables par Basemap :
    liste_coord = []
    for i in df_coord:
        if i.any() != "": # On ne prend pas en compte les lignes vides
            long = get_longitude(i[0])
            lat = get_latitude(i[1])
            liste_coord.append([long, lat])
    
    
    # On plot avec Basemap :
    ax.clear()

    m = Basemap(width=12000000,height=9000000,projection='lcc',
                resolution='c',lat_0=liste_coord[0][1],lon_0=liste_coord[0][0]) # On centre la carte sur la position du 1er
    m.bluemarble(scale=0.5);
    m.etopo(scale=0.5, alpha=0.5)
    m.drawcountries(linewidth=0.2)
    
    # Double boucle pour les différentes dates ! Etape d'après : afficher les noms et assigner une couleur.
    # N'afficher que le nom du premier !
    # Plot points : m(long, lat)
    
    for i in liste_coord:
        if i == liste_coord[0]:
            x,y = m(i[0], i[1])
            ax.plot(x, y, 'd', markersize=15)
            ax.text(x, y, df_firstskipper, fontsize=15)
        else :
            x,y = m(i[0], i[1])
            ax.plot(x, y, 'd', markersize=15)

    plt.title("Historique de la course - Vendée Globe 2020", size=30)
    plt.box(False)
    plt.tight_layout()
    


# In[26]:


# Chart race. On affiche que les 7 premiers de la course !
fig, ax = plt.subplots(figsize=(15, 8))
animator = animation.FuncAnimation(fig, draw_chartrace, frames=liste_dates, interval=1500, repeat=False)
HTML(animator.to_jshtml())


# ## Représentation de la course avec Basemap :
# 

# In[83]:


# Fonction pour représenter la course sur une carte du monde, aujourd'hui !
#On utilise ici la projection de lambert :

geoplot_ajd_lambert()


# In[86]:


# Fonction pour représenter la course sur une carte du monde, aujourd'hui !
#On utilise ici la projection sinusoidale :
geoplot_ajd_sinu()


# In[87]:


############## PROJECTION DYNAMIQUE DE LA COURSE DEPUIS LE DEBUT DU VENDEE GLOBE
#### ATTENTION : le temps de chargement de la séquence peut mettre jusqu'à 1min30

%%time
matplotlib.rcParams['animation.embed_limit'] = 2**128
fig, ax = plt.subplots(figsize=(15,10))
animator = animation.FuncAnimation(fig, geoplot, frames=liste_dates, interval=900, repeat=False)
HTML(animator.to_jshtml())


# # Partie 2 : Analyses et corrélations

# Présenter les corrélations, montrer la pertinence de faire des corrélations (on compare pas des choux et des carrottes), pourquoi est ce qu'on peut comparer tel paramètre et qu'est ce qu'on peut pas faire

# In[30]:


# Corrélation entre classement et VMG des voiliers :

df_corr = df_classement.reset_index()
df_corr = df_corr[["Classement", "VMG"]]
df_corr = df_corr.dropna()

print("Matrice de corrélation :")
print(df_corr.corr(method="pearson"))

sns.pairplot(df_corr)


# Commentaire : plus la VMG est grande, plus la position est petite !
# On peut chercher à faire une regression linéaire pour expliquer le classement en fonction de la VMG :

# In[31]:


# Régression linéaire entre la VMG et la position :
model = LinearRegression(fit_intercept=True)
X = np.array(df_corr["Classement"])
X1 = np.c_[np.ones(len(X)),X]
Y = np.array(df_corr["VMG"])
model.fit(X1,Y)
Y_predict = model.predict(X1)

print("Coefficient de détermination du modèle linéaire :", model.score(X1,Y))

plt.figure(figsize=[10,6])
plt.scatter(X,Y,color='b')
plt.xlabel("Position des skippers")
plt.ylabel("VMG des skippers")
plt.plot(X,Y_predict, color='r', linewidth=7)


# Le coefficient de détermination ("score") de ce modèle n'est pas bon : il est de **R² = 0.16**.
# Instinctivement, la relation entre la VMG et le classement devrait être linéaire. On observe ici une légère tendence mais rien de plus.
# On peut dès lors se poser la question : parmis tous les paramètres accumulés dans les DataFrames précédents, lequel explique le mieux le fait qu'un Skipper soit premier de la course ?
# 
# #### On pourra aussi regarder si la différence dans le classement entre 2 dates est bien dû à une différence de vitesse / VMG ! 
# ### Corréler classement avec distance totale parcourue

# **On peut regarder quel paramètre influe le + sur le fait d'être dans les 3 premiers :**
# ### Lister d'autres caractéristiques : distance totale parcourue, influence du bateau sur la route choisie, est ce que celui qui parcoure + de distance est mieux classé ?
# 
# ### Listons les paramètres qui peuvent influer :
# 

# In[32]:


# Dans une fonction :
liste_param = ["Déplacement (poids)", "Presence foils", "Surface de voiles au près", "Surface de voiles au portant", "Largeur", "Tirant d'eau"]
df_influence = df_merged[["Skipper", "Classement"]+liste_param]
df_influence["Top3"] = df_influence["Classement"].apply(isintop3)
df_influence.dropna(axis=0, inplace=True)
X = df_influence[liste_param]
Y = df_influence["Top3"].to_numpy()


# Random Forest Classifier test :
modelRFC = RandomForestClassifier(n_estimators=100)
modelRFC.fit(X,Y)
y_RFC = modelRFC.predict(X)
print("Le score du modèle de Random Tree Classifier est de :",modelRFC.score(X,Y))

dfeatures = pd.DataFrame(zip(modelRFC.feature_importances_, liste_param), columns=["Importance", "Paramètre"])
dfeatures = dfeatures.sort_values(["Importance"], ascending=False)

plt.figure(figsize=[12,7])
sns.barplot(x="Importance", y="Paramètre", data=dfeatures, palette="Spectral")


# ### Ainsi, nous constatons que les paramètres qui influent le plus sur les performances (pour être dans le top 3) sont les surfaces des voiles, suivi du poids du bateau.
# ### La présence de foils a également une légère influence !
