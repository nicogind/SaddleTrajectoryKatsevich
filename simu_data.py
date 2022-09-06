#author: Nicolas Gindrier
#This file produces cone beam data

import numpy as np
import itk
from itk import RTK as rtk
dtype = np.float32

import matplotlib.pyplot as plt
# Define the image type, we will use 3D float images in these hands-on sessions
ImageType = itk.Image[itk.F,3]

# choice of the phantom
# 1 ellipse
# 2 Nico phantom
# 3 thorax

# par defaut
phantom = 2
nprojections = 360
draw = True
spacing = 2
D = 400 
R = 250.
H = 100.
Ldet = 1000 # largeur detecteur
Hdet = 400 # hauteur detecteur

# lire config.txt pour avoir les parametres
file = open('config.txt',"r")
line = file.readline()
while line:
    a=line.split()
    line = file.readline()
    if len(a)>=2:
        a0=a[0]
        a1=int(a[1])
        if(a0=="phantom"):
            phantom=a1
        elif(a0=="nprojections"):
            nprojections=a1
        elif(a0=="draw"):
            draw = a1
        elif(a0=="spacing"):
            spacing = a1   
        elif(a0=="D"): # distance source detecteur
            D = a1
        elif(a0=="R"): # rayon trajectoire
            R = a1 
        elif(a0=="H"):# hauteur trajectoire
            H = a1
        elif(a0=="Hdet"): # hauteur detecteur
            Hdet = a1
        elif(a0=="Ldet"): # largeur detecteur
            Ldet = a1

ell=[60,20,20]
#ell=[30,10,10]

if draw:
    if phantom==1:
        ellipsoid = rtk.ConstantImageSource[itk.Image[itk.F,3]].New(Size=[256]*3,Spacing=[1]*3,Origin=[-128]*3,Constant=0)
        ellipsoid = rtk.DrawEllipsoidImageFilter(Input=ellipsoid.GetOutput(), Axis=ell, Center=[0,0,0], Density=2)
        itk.imwrite(ellipsoid,"ellipsoid.mha") 
        print("ellipsoid.mha ecrit")
    elif phantom==2:
    # origine = -size * spacing / 2
        sldraw = rtk.ConstantImageSource[itk.Image[itk.F,3]].New(Size=[135,135,45],Spacing=[1]*3,Origin=[0]*3,Constant=0)
        sldraw = rtk.DrawGeometricPhantomImageFilter(Input=sldraw.GetOutput(), ConfigFile="NicoPhantom.txt",IsForbildConfigFile=True,PhantomScale=1,OriginOffset=[67,67,22])
        itk.imwrite(sldraw,"NicoPhantom.mha")  
        print("NicoPhantom.mha ecrit")
    elif phantom==3:
        # fantome Thorax avec cylindres
        sldraw = rtk.ConstantImageSource[itk.Image[itk.F,3]].New(Size=[190,76,191],Spacing=[2]*3,Origin=[-189,-75+ycenter,-190],Constant=0)
        sldraw = rtk.DrawGeometricPhantomImageFilter(Input=sldraw.GetOutput(), ConfigFile="thorax_full.txt",IsForbildConfigFile=True,PhantomScale=7.5,OriginOffset=[0,ycenter/7.5,0])
        itk.imwrite(sldraw,"Thorax.mha") 
        print("Thorax.mha ecrit")
    elif phantom==4:
        sldraw = rtk.ConstantImageSource[itk.Image[itk.F,3]].New(Size=[190,76,191],Spacing=[2]*3,Origin=[-189,-75+ycenter,-190],Constant=0)
        sldraw = rtk.DrawGeometricPhantomImageFilter(Input=sldraw.GetOutput(), ConfigFile="thorax_full2.txt",IsForbildConfigFile=True,PhantomScale=7.5,OriginOffset=[0,ycenter/7.5,0])
        itk.imwrite(sldraw,"ThoraxSC.mha") 
        print("ThoraxSC.mha ecrit")
# Define geometry
geometry = rtk.ThreeDCircularProjectionGeometry.New()
lam = np.linspace(0,2*np.pi-2*np.pi/nprojections,nprojections)
#lam = np.linspace(-np.pi,np.pi-2*np.pi/nprojections,nprojections)

for l in lam:
    cl = np.cos(l)
    sl = np.sin(l)
    srcOffX = R * cl
    srcOffY = R * sl
    srcToIso = H * np.cos(2*l)
    Sl = np.array([srcOffX,srcOffY,srcToIso])
    nSl = np.linalg.norm(Sl)
    # coordonnees detecteur : detecteur droit
    Xd = Sl * (1-D / R)
    projOffX = Xd[0]
    projOffY = Xd[1]
    Xd[2] = H * np.cos(2*l) # ici le detecteur est a la meme hauteur que la source
    #projOffZ = srcToIso * np.linalg.norm(Xd) / R
    projOffZ = Xd[2]
    detCol = [0,0,1]
    detRow = [-sl,cl,0]

    geometry.AddProjection(Sl,[projOffX,projOffY,projOffZ],detRow,detCol)
       
geometrywriter = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
geometrywriter.SetFilename("geometry.xml")
geometrywriter.SetObject(geometry)
geometrywriter.WriteFile()

# dimensions detecteur
srcproj = rtk.ConstantImageSource[ImageType].New()
# ld et nSL dependent de lambda mais ld / nSl presque constant en pratique
#Hdet = D * Hfov / nSl # calcul approximatif mais on tronque apres
Hfov = Hdet / D * nSl
srcproj.SetSize([int(Ldet / spacing), int(Hdet / spacing) , int(nprojections)])
srcproj.SetSpacing([spacing]*3)
srcproj.SetOrigin([-Ldet / 2.0,-Hdet / 2.0,-nprojections/2])

projname = './proj/proj'+str(nprojections)

if phantom==1:
    # projections
    #Axis=[80,70,60] fonctionne, mais pas forcement plus petit (anormal)
    proj = rtk.RayEllipsoidIntersectionImageFilter(Input=srcproj.GetOutput(), Geometry=geometry, Axis=ell, Center=[0,0,0], Density=2)
    #sinogram = itk.GetArrayFromImage(proj).squeeze()
    projname = projname + 'Elli.mha'
    itk.imwrite(proj,projname) 
#    # retroprojection dans un volume de 1 (fov)
#    srcproj.SetConstant(1.)
#    fov = rtk.BackProjectionImageFilter[ImageType, ImageType].New()
#    fov.SetGeometry(geometry)
#    fov.SetInput(srcvol.GetOutput())
#    fov.SetInput(1, srcproj.GetOutput())
#    itk.imwrite(fov.GetOutput(),"fov.mha") 
#
elif phantom==2:
    projNico = rtk.ProjectGeometricPhantomImageFilter(Input=srcproj.GetOutput(), ConfigFile="NicoPhantom.txt",IsForbildConfigFile=True,OriginOffset=[0,0,0], Geometry=geometry,PhantomScale=1)
    projname = projname + 'NicoPhantom.mha'
    itk.imwrite(projNico,projname) 

if phantom==3:
    projThorax = rtk.ProjectGeometricPhantomImageFilter(Input=srcproj.GetOutput(), ConfigFile="thorax_full.txt",IsForbildConfigFile=True,OriginOffset=[0,ycenter/7.5,0], Geometry=geometry,PhantomScale=7.5)
    projname = projname + 'Thorax.mha'
elif phantom==4:
    projThorax = rtk.ProjectGeometricPhantomImageFilter(Input=srcproj.GetOutput(), ConfigFile="thorax_full2.txt",IsForbildConfigFile=True,OriginOffset=[0,ycenter/7.5,0], Geometry=geometry,PhantomScale=7.5)
    projname = projname + 'ThoraxSC.mha'



print(projname+ " ecrit")
   # FOV
srcproj.SetConstant(1.)
## volume pour la retroprojection
srcvol = rtk.ConstantImageSource[ImageType].New()
srcvol.SetConstant(0.) # Note that this is useless because 0 is the default
srcvol.SetSize([256]*3)
srcvol.SetSpacing([1]*3)
srcvol.SetOrigin([-128]*3)
#
fov = rtk.BackProjectionImageFilter[ImageType, ImageType].New()
fov.SetGeometry(geometry)
fov.SetInput(srcvol.GetOutput())
fov.SetInput(1, srcproj.GetOutput())
#
#if(tronc):
#    crop_fov = itk.CropImageFilter[type(srcproj.GetOutput()),type(srcproj.GetOutput())].New()
#    crop_fov.SetInput(srcproj)
#    crop_fov.SetLowerBoundaryCropSize(tronc_low)
#    crop_fov.SetUpperBoundaryCropSize(tronc_up)
#    fov.SetInput(1, crop_fov.GetOutput())
#    nameFov = "fov_tronc_case"+str(tronc)+".mha"
#else:
#    fov.SetInput(1, srcproj.GetOutput())
#    nameFov = "fov.mha"
itk.imwrite(fov.GetOutput(),"fov.mha")
print("fov.mha ecrit")

