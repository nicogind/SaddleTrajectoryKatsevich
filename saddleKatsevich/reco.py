#author: Nicolas Gindrier
#This file reconstructs the image with the Katsevich algorithm (Exact cone beam reconstruction for a saddle trajectory, Haiquan Yang, Meihua Li, Kazuhito Koizumi and Hiroyuki Kudo, 2006)

import numpy as np
import itk
from itk import RTK as rtk
import time
import os
from matplotlib import pyplot as plt
from scipy import signal
import scipy.signal
import math
import copy

dtype = np.float32

start = time.time()
#valeurs par defaut
nprojections = 360
# lire config.txt
file = open('config.txt',"r")
line = file.readline()
while line:
    a=line.split()
    line = file.readline()
    if len(a)>=2:
        a0=a[0]
        a1=int(a[1])
        if(a0=="nprojections"):
            nprojections=a1
        elif(a0=="phantom"):
            phantom=a1 
        elif(a0=="D"):
            D = a1
        elif(a0=="R"):
            R = a1
        elif(a0=="spacing"):
            spacing = a1
        elif(a0=="H"):
            H = a1 
        elif(a0=="Hdet"): # hauteur detecteur
            Hdet = a1
        elif(a0=="Ldet"): # largeur detecteur
            Ldet = a1

# DIMENSIONS FOV
h1 = 30 #approximatif, on donne des dimensions plus petites, 44 est la vraie dimension 
r=90#100 est la vraie dimension

name = '/kat'+str(nprojections)+'proj_'
projname = './proj/proj'+str(nprojections)
if phantom==1:
    projname += 'Elli.mha'
    name += 'Elli'
elif phantom==2:
    projname += 'NicoPhantom.mha'
    name += 'NicoPhantom'
elif phantom==3:
    projname += 'Thorax.mha'
elif phantom==4:
    projname += 'ThoraxSC.mha'
proj = itk.imread(projname)

nrep=0
ok=False
while not ok:
    rep = "./reco2"+name+str(nrep)+".mha"
    if not os.path.exists(rep):
         ok=True
         print(rep+" cree")
    else:
        nrep+=1
name = name+str(nrep)

#DEBUT DE L IMPLEMENTATION
#----------------
d_lam = 2 * np.pi / nprojections
g_f = itk.GetArrayFromImage(proj)
proj_size = np.array(np.shape(g_f)) # ATTENTION A L ORDRE
n_lam = proj_size[0]
n_w = proj_size[1]
n_u = proj_size[2]
#half shift pixel
#-------------------
Lam = np.linspace(d_lam/2,2*np.pi-d_lam/2,nprojections-1)
u = np.arange(-Ldet/2+spacing/2,Ldet/2-spacing/2,spacing)
w = np.arange(-Hdet/2+spacing/2,Hdet/2-spacing/2,spacing)

u_grid = u * np.ones((proj_size-1)) #on repete u "dans les lignes"
w_grid = np.array([np.reshape(np.repeat(w,n_u-1),[n_w-1,n_u-1]) for i in range(n_lam-1)]) # on repete w "dans les colonnes"

# CALCUL DE G1 (DERIVEE)
# ---------------------
g1lam = g_f[1:n_lam,:n_w-1,:n_u-1]- g_f[:n_lam-1,:n_w-1,:n_u-1] \
    +g_f[1:n_lam,:n_w-1,1:n_u]- g_f[:n_lam-1,:n_w-1,1:n_u]\
    +g_f[1:n_lam,1:n_w,1:n_u]- g_f[:n_lam-1,1:n_w,1:n_u] \
    +g_f[1:n_lam,1:n_w,:n_u-1]- g_f[:n_lam-1,1:n_w,:n_u-1] 
g1lam = g1lam / (4*d_lam)
 
g1w =  g_f[:n_lam-1,1:n_w,:n_u-1]- g_f[:n_lam-1,:n_w-1,:n_u-1] \
    +g_f[:n_lam-1,1:n_w,1:n_u]- g_f[:n_lam-1,:n_w-1,1:n_u]\
    +g_f[1:n_lam,1:n_w,1:n_u]- g_f[1:n_lam,:n_w-1,1:n_u] \
    +g_f[1:n_lam,1:n_w,:n_u-1]- g_f[1:n_lam,:n_w-1,:n_u-1] 
g1w = g1w / (4*spacing)

g1u = g_f[:n_lam-1,:n_w-1,1:n_u]- g_f[:n_lam-1,:n_w-1,:n_u-1] \
    +g_f[:n_lam-1,1:n_w,1:n_u]- g_f[:n_lam-1,1:n_w,:n_u-1]\
    +g_f[1:n_lam,1:n_w,1:n_u]- g_f[1:n_lam,1:n_w,:n_u-1] \
    +g_f[1:n_lam,:n_w-1,1:n_u]- g_f[1:n_lam,:n_w-1,:n_u-1] 
g1u = g1u / (4*spacing)

g_1 = g1lam + ((u_grid**2 + D**2) / D) * g1u + (u_grid * w_grid / D) * g1w


print("Calcul de g1 fait")

# CALCUL DE G2 (WEIGHTING)
# ---------------------
g_2 = g_1 * D / np.sqrt(D**2+u_grid**2+w_grid**2) 
print("Calcul de g2 fait")

# CALCUL DE G3 
# REARRANGEMENT FILTERING LINES
#-----------------------
def w_hat(lam,z_hat,u):
    if z_hat < 0:
        a = np.tan(lam)
    else: # pour z_hat > 0 lam ne peut pas valoir 0 ou pi
        a = -1. / np.tan(lam)
    return (u * a + D) / R * z_hat

# donne l'indice de w
def w_to_iw(W):
    return int((len(w)-1) * (W+Hdet/2-spacing/2) / (Hdet-spacing)) #par defaut

g_3 = np.zeros((proj_size-1)) # g_3(lam,z_hat,u)

# affiche la progression du calcul
def progress(i,n):
    if 100*i/n-100*int(100*i/n)<2./n:
        print(100*i/n," %")

# CALCUL DES Z HAT (PARAMETRE DES PLANS FILTRANTS)
def w_bar_min(lam):
    a = H*np.cos(2*lam)
    if a < -h1:
        return -D * (h1+a) / (R+r)
    else:
        return -D * (h1+a) / (R-r)

u_tilde = r * D / np.sqrt(R**2-r**2)
#u_tilde = u[-1]
def w_bar_max(lam):
    a = H*np.cos(2*lam)
    if a > h1:
        return D * (h1-a) / (R+r)
    else:
        return D * (h1-a) / (R-r)
def z_hat_min(lam):
    a = H*np.cos(2*lam)
    if a < -h1:
        return w_bar_min(lam) * R * np.sin(lam) / (D * np.sin(lam)+np.sign(np.tan(lam))*np.cos(lam)*u_tilde)
    else:
        return w_bar_min(lam) * R * np.cos(lam) / (D * np.cos(lam)-np.sign(np.tan(lam))*np.sin(lam)*u_tilde)

def z_hat_max(lam):
    a = H*np.cos(2*lam)
    if a > h1:
        return w_bar_max(lam) * R * np.cos(lam) / (D * np.cos(lam)+np.sign(np.tan(lam))*np.sin(lam)*u_tilde)
    else:
        return w_bar_max(lam) * R * np.sin(lam) / (D * np.sin(lam)-np.sign(np.tan(lam))*np.cos(lam)*u_tilde)

i = 0
print("Calcul des zhat")
for LAM in Lam:
    k = 0
    progress(i,len(Lam))
    Z_hat = np.linspace(z_hat_min(LAM),z_hat_max(LAM),n_w-1)
    for U in u:
        if np.abs(U)<u_tilde:
        #interpolation 
            j = 0
            for z_hat in Z_hat:
                jw = w_to_iw(w_hat(LAM,z_hat,U))
                g_3[i,j,k] = g_2[i,jw,k]
                j += 1
        k += 1
    i += 1

print("Calcul de g3 fait")
# CALCUL DE G4
# TRANSFORMEE DE HILBERT
#-----------------------
th = 3 # 1 Fourier sans halft shift 2 Fourier avec halft shift 3 methode trapeze (Defrise)
Ham = False # filtrage de Hamming
g_4 = np.zeros((proj_size-1))
if th<3: #methode de Fourier
    # kbl : noyau de Hilbert modifie
    #k=np.zeros(len(kbl))
    #k[-1]=1
    if th==1:
        u_prime = spacing*(np.arange(-n_u+1,n_u,1)) #TEST MEILLEUR
        kbl = 1-np.cos(np.pi*u_prime/spacing)#=[2,0,2,0,...,0,2]
        nz = np.where(kbl!=0) #[-n_u+1,...,-10,-8,-6,...,n_u]
        kbl[nz]=kbl[nz]/(np.pi*u_prime[nz]) #2/(j pi) pour j pair 
        kbl *= spacing
    elif th==2: # fourier avec half shift, 
        u_prime = np.arange(-Ldet+spacing/2,Ldet-spacing/2,spacing) 
        kbl = 1-np.cos(np.pi*u_prime/spacing)
        nz = np.where(kbl!=0) 
        kbl[nz]=kbl[nz]/(np.pi*u_prime[nz])  
        kbl *= spacing
    if Ham:
        ham=scipy.signal.get_window("hamming",np.size(kbl))
        #kbl = signal.convolve(ham,kbl,mode='same')
        kbl *= ham
    for i in range(n_lam-1):
        for j in range(n_w-1): #convolution
            g_4[i,j,:] = signal.convolve(g_3[i,j,:],kbl,mode='same')
elif th == 3:  
    u_prime = np.arange(-Ldet/2,Ldet/2-spacing,spacing)
    for k in range(n_u-1):
        for i in range(n_u-1): 
            g_4[:,:,k] += g_3[:,:,i]*spacing / (np.pi*(u_prime[k]-u[i])) 

print("Calcul de g4 fait")

def u_to_iu(u):
    if th==1:
        return int(round((n_u-1) * (u-spacing/2+Ldet/2) / (Ldet-spacing))) #sans half shift
    else:
        return int(round((n_u-1) * (u+Ldet/2) / (Ldet-spacing))) # avec half shift

# 2e REARRANGEMENT 
print("Début du réarrangement")
g_F = np.zeros((proj_size-1))
i = 0
for l in Lam:
    progress(i,len(Lam))
    zhmin=z_hat_min(l)
    zhmax=z_hat_max(l)
    w_min= w_bar_min(l)
    w_max= w_bar_max(l)
    iu = 0
    for U in u:
        if np.abs(U)<u_tilde:
            iw = 0
            for W in w:
                if W > w_min and W < w_max:
                    if W >0:
                        z_hat = W * R * np.sin(l) / (D * np.sin(l) - U * np.cos(l))
                    else:
                        z_hat = W * R * np.cos(l) / (D * np.cos(l) + U * np.sin(l))
                    iz = int((len(w)-1) * (z_hat-zhmin) / (zhmax-zhmin))
                    g_F[i,iw,iu] = g_4[i,iz,iu]
                iw += 1
        iu += 1
    i += 1
gF_image = itk.GetImageFromArray(np.float32(g_F))    
gF_image.SetOrigin([-n_u/2,-n_w/2,-n_lam/2])
itk.imwrite(gF_image,"gF7.mha")
print("Calcul de gF fait")

# RETROPROJECTION
#-----------------------
def v_bar(x1,x2,lam):
    return R-x1*np.cos(lam)-x2*np.sin(lam)

def u_bar(x1,x2,lam):
    a = D / v_bar(x1,x2,lam)
    return  a * (-x1 * np.sin(lam) + x2*np.cos(lam))

def w_bar(x,lam):
    a = D / v_bar(x[0],x[1],lam)
    return a * (x[2] - H * np.cos(2*lam))

# backprojection
def f(x):
    s = 0
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    il = 0
    for l in Lam:
        iub =  u_to_iu(u_bar(x1,x2,l))
        iwb = w_to_iw(w_bar(x,l))
        s += g_F[il,iwb,iub] / v_bar(x1,x2,l)
        il += 1
    s *= d_lam / (4*np.pi)
    return s

x = np.zeros(3)
x_size = int(1.5*r)
y_size = int(1.5*r)
#z_size = int(H/2)
z_size = int(1.5*h1)
print("size",x_size,z_size) #A SUPPRIMER
reco_spacing = 1
origin = np.array([-int(x_size/2),-int(y_size/2),-int(z_size/2)])#par defaut sans le int
reco = np.zeros((z_size,y_size,x_size))# ATTENTION ORDRE
x = np.zeros(3)
# compute each point
print("Debut de la retroprojection")
for i in range(x_size):
    progress(i,x_size-1)
    for j in range(y_size):
        for k in range(z_size):
            x[0] = (i*reco_spacing)
            x[1] = (j*reco_spacing)
            x[2] = (k*reco_spacing)
            x+=origin
            reco[k,j,i] = f(x) # ATTENTION ORDRE
#ECRITURE FICHIER
#----------------
reco_image = itk.GetImageFromArray(np.float32(reco))
end = time.time()
print("temps de reconstruction "+str(end-start)+" s")
itk.imwrite(reco_image,"./reco2"+name+".mha")
print(name+" ecrit")
