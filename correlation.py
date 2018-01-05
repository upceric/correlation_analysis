# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:07:48 2016
@author: lww
"""
import pyvoro
import scipy.spatial as sptl
from matplotlib import pyplot as plt 
import numpy as np
from matplotlib import collections as mc
filename='1000_50p_1_4_bidi.dat'
num_lines = sum(1 for line in open(filename))
print("total number of lines: "+str(num_lines))
R_1=max(float(i.strip().split()[2]) for i in open(filename))/2
R_2=min(float(i.strip().split()[2]) for i in open(filename))/2

print("R_1:"+str(R_1))
print("R_2:"+str(R_2))

circles=np.zeros((num_lines*9,3))
np.random.seed(11)

i=0
coef=1
species=[]
with open(filename,"r") as filestream:
    for line in filestream:
        cl=line.split()
        species+=[cl[3]]*9
        circles[i,0]=float(cl[0])
        circles[i,1]=float(cl[1])
        circles[i,2]=float(cl[2])/2*coef
        i+=1
        circles[i,0]=float(cl[0])
        circles[i,1]=float(cl[1])+1
        circles[i,2]=float(cl[2])/2*coef
        i+=1
        circles[i,0]=float(cl[0])+1
        circles[i,1]=float(cl[1])+1
        circles[i,2]=float(cl[2])/2*coef
        i+=1
        circles[i,0]=float(cl[0])-1
        circles[i,1]=float(cl[1])+1
        circles[i,2]=float(cl[2])/2*coef
        i+=1
        circles[i,0]=float(cl[0])
        circles[i,1]=float(cl[1])-1
        circles[i,2]=float(cl[2])/2*coef
        i+=1
        circles[i,0]=float(cl[0])-1
        circles[i,1]=float(cl[1])-1
        circles[i,2]=float(cl[2])/2*coef
        i+=1
        circles[i,0]=float(cl[0])+1
        circles[i,1]=float(cl[1])-1
        circles[i,2]=float(cl[2])/2*coef
        i+=1
        circles[i,0]=float(cl[0])+1
        circles[i,1]=float(cl[1])
        circles[i,2]=float(cl[2])/2*coef
        i+=1
        circles[i,0]=float(cl[0])-1
        circles[i,1]=float(cl[1])
        circles[i,2]=float(cl[2])/2*coef
        i+=1

radius=circles[:,2].tolist()
circles_input=circles[:,0:2].tolist()
#radius=[0.1,0.1]
#circles_input=[[0.3,0.6],[0.7,0.3]]
x_box=[-1, 2]
y_box=[-1, 2]
cells = pyvoro.compute_2d_voronoi(
  circles_input, # point positions, 2D vectors this time.
  [x_box, y_box], # box size, again only 2D this time.
  0.1, # block size; same as before.
  radii=radius # particle radii -- optional and keyword-compatible.
)
center=[ii['original'] for ii in cells]
polyline=[]
triangle=[]
volume=[]
volume_radius=[]#there are nine copies of the matrix, we need to record the radii for what we are interested
id_1=[]#record the cell id 
speci_rec=[]#record the species
visited=[]#record the grid is visited or not
ii=0
for i in cells:
    if center[ii][0]>0 and center[ii][0]<1 and center[ii][1]>0 and center[ii][1]<1:
        for j in range(0,len(i['vertices'])):
            polyline=polyline+[(i['vertices'][j%len(i['vertices'])],i['vertices'][(j+1)%len(i['vertices'])])]
        for j in range(0,len(i['faces'])):
            if i['faces'][j]['adjacent_cell']>=0:
                triangle=triangle+[([i['original'],cells[i['faces'][j]['adjacent_cell']]['original']])]
        volume.append(i['volume'])
        volume_radius.append(radius[ii])
        id_1.append(ii)
        speci_rec.append(species[ii])
        visited.append(False)
    i['id']=ii
    ii+=1
poly_lc=mc.LineCollection(polyline,color='k',linewidth=0.4,linestyle='solid')
tri_lc=mc.LineCollection(triangle,color='k',linewidth=0.4,linestyle='solid')
plt.gca().add_collection(poly_lc)#adding the edges of triangles
#plt.gca().add_collection(tri_lc)#adding the edges of triangles
'''
new_tri=sptl.Delaunay(circles_input)
for t in new_tri.simplices:
    t_i=[t[0],t[1],t[2],t[0]]
    x=[circles_input[i][0] for i in t_i]
    y=[circles_input[i][1] for i in t_i]
    #plt.plot(x,y,color='k')
'''
for j in range(0,len(cells)):
    #if circles[j,2]==R_1/2*0.8:
    #    color=np.array([0.000, 1.0, 0.000])
    #else:
    color=np.array([0.00, 0.0, 0.000])
    if center[j][1]<0:
        color=np.array([1.000, 0.0, 0.0])
    if center[j][1]>1:
        color=np.array([0.000, 0.0, 1.0])
    if center[j][0]<0:
        color=np.array([1.000, 0.000, 0.000])
    if center[j][0]>1:
        color=np.array([1.000, 0.000, 0.000])
    if center[j][0]>0 and center[j][0]<1 and center[j][1]>0 and center[j][1]<1:
        if species[j]=="1":
           circle=plt.Circle((center[j][0],center[j][1]),radius=radius[j],color=color,fill=True,alpha=1)
        else:
           circle=plt.Circle((center[j][0],center[j][1]),radius=radius[j],color=color,fill=True,alpha=0.5)
    else:
        circle=plt.Circle((center[j][0],center[j][1]),radius=radius[j],color=color,fill=True,alpha=0.5)
    plt.gca().add_patch(circle)
volume_norm=[i/(np.pi*(R_2*R_2)) for i in volume]
free_volume=[]
free_volume_1=[]
free_volume_2=[]
for idx,val in enumerate(volume):
    free_volume.append(val-np.sqrt(3)*2*volume_radius[idx]*volume_radius[idx])
    if speci_rec[idx]=='1':
        free_volume_1.append(val-np.sqrt(3)*2*volume_radius[idx]*volume_radius[idx])
    else:
        free_volume_2.append(val-np.sqrt(3)*2*volume_radius[idx]*volume_radius[idx])
free_volume_norm=[i/(np.pi*(R_2*R_2)) for i in free_volume]
free_volume_1=[i/(np.pi*(R_2*R_2)) for i in free_volume_1]
free_volume_2=[i/(np.pi*(R_2*R_2)) for i in free_volume_2]
print("haha",len(free_volume_norm),len(volume_radius),len(free_volume))
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-0,1)
plt.ylim(-0,1)
plt.axis('off')

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon as SPolygon
from matplotlib.patches import Polygon
def find_point_in_polygon_index(cells=None,point=None,cc='r'):
    '''
    loop all the cells and find the polygon that contains
    the point
    '''
    for i in cells:
        polygon=SPolygon(i['vertices'])
        if polygon.contains(point):
            #print("find")
            axes=plt.gca()
            axes.add_patch(Polygon(i['vertices'],closed=True,\
                                   facecolor=cc,alpha=0.3))
            #print(i)
def plot_polygon_with_id(cells=None,all_id=None,cc='r'):
    ii=0
    for i in all_id:
        if ii%2==0:
            cc='r'
        else:
            cc='g'
        ii+=1
        axes=plt.gca()
        axes.add_patch(Polygon(cells[i]['vertices'],closed=True,\
                                   facecolor=cc,alpha=0.6))        
def find_point_in_polygon_p(cells=None,point=None):
    '''
    loop all the cells and find the polygon that contains
    the point,with printing the center
    '''
    for i in cells:
        polygon=SPolygon(i['vertices'])
        if polygon.contains(point):
            #print("find")
            axes=plt.gca()
            axes.add_patch(Polygon(i['vertices'],closed=True,\
                                   facecolor='b',alpha=0.2))
            print(i)
def return_center(cells=None,point=None):
    '''
    loop all the cells and find the polygon that contains
    the point,with printing the center
    '''
    for i in cells:
        polygon=SPolygon(i['vertices'])
        if polygon.contains(point):
            #print("find")
            return i['original']
#point=Point(0.5,0.5)
def plot_point_with_circles(cells=None,point=None):
    '''
	plot the central point with rings
	'''
    orig=return_center(cells,point)
    orig=orig[0:2]
    L=0.2
    for i in range(0,360,1):
        x=orig[0]+L*np.cos(i)
        y=orig[1]+L*np.sin(i)
        find_point_in_polygon_index(cells,Point(x,y),cc='r')
        x=orig[0]+2*L*np.cos(i)
        y=orig[1]+2*L*np.sin(i)
        find_point_in_polygon_index(cells,Point(x,y),cc='g')
    
    circle=plt.Circle((orig[0],orig[1]),radius=L,color='r',fill=False,alpha=1)
    plt.gca().add_patch(circle)
    circle=plt.Circle((orig[0],orig[1]),radius=L*2,color='g',fill=False,alpha=1)
    plt.gca().add_patch(circle)
    find_point_in_polygon_p(cells,Point(orig))

#plot_point_with_circles(cells,Point(0.1,0.1))
#plt.show()
###################################calculate the correlation
def find_index(cells=None,point=None,cc='r'):
    '''
    based on the point, find the cells id
    '''
    for i in cells:
        polygon=SPolygon(i['vertices'])
        if polygon.contains(point):
            return(i['id'])

def construct_pair(cells=None,point=None,L=None,id_pair=None):
    '''
	plot the central point with rings
	'''
    orig=return_center(cells,point)
    orig=orig[0:2]
    id_ori=find_index(cells,point)
    import random
    val=random.sample(range(0,360),1)
    #for i in range(0,360,72):
    for i in range(0,360,72):
        x=orig[0]+L*np.cos(np.deg2rad(i+float(val[0])))
        y=orig[1]+L*np.sin(np.deg2rad(i+float(val[0])))
        id_pa=find_index(cells,Point(x,y),cc='r')
        if (id_pa in id_1) and (id_pa!=id_ori):
            #find_point_in_polygon_index(cells,Point(x,y),cc='r')
            if np.random.uniform(0,1)>0.5:
                id_pair.append([id_ori,id_pa])
            else:
                id_pair.append([id_pa,id_ori])
            break                

def remove_repitition(a=None,free_volume_norm=None):
    '''
    remove the pairs which are in common, including one common cell
    '''
    b=np.zeros(len(a),dtype=np.int)+1
    for i,j in enumerate(a):
        for k in range(i+1,len(a)):
            #if set(a[i])==set(a[k]):
            #removing all the pair which have commom
            if len(set(a[i])-(set(a[i])-set(a[k])))>0:
                b[i]=0
                continue
    from itertools import compress
    aa=[]
    bb=[]#species array
    #print("aa:",a)
    for i in a:
        #print(id_1.index(i[0]),id_1.index(i[1]))
        aa.append([free_volume_norm[id_1.index(i[0])],\
                    free_volume_norm[id_1.index(i[1])]])
        bb.append([speci_rec[id_1.index(i[0])],speci_rec[id_1.index(i[1])]])
    
    remove_share=False
    if remove_share:
        c=list(compress(aa,b))
        bb_c=list(compress(bb,b))
    else:
        c=aa
        bb_c=bb
    
    d=[]#all the id
    for i in list(compress(a,b)):
        d=d+i
    #plot_polygon_with_id(cells,all_id=d,cc='r')
    #print(list(compress(a,b)))
    v_i=[i[0] for i in c]
    v_j=[i[1] for i in c]
    m_v_i=np.mean(v_i)
    m_v_j=np.mean(v_j)
    sigma_i_s=np.std(v_i)**2#sigma of v_i array
    sigma_j_s=np.std(v_j)**2#sigma of v_j array
    correlation_p=sum([(i-m_v_i)*(j-m_v_j) for i,j in zip(v_i,v_j)])\
            /np.sqrt(sum([(i-m_v_i)**2 for i in v_i])*(sum([(j-m_v_j)**2 for j in v_j])))
    correlation=sum([(i-m_v_i)*(j-m_v_j) for i,j in zip(v_i,v_j)])\
                /(sigma_i_s+sigma_j_s)*2/len(v_i)
    
    v_i_bb=[float(i[0]) for i in bb_c]
    v_j_bb=[float(i[1]) for i in bb_c]
    m_v_i_bb=np.mean(v_i_bb)
    m_v_j_bb=np.mean(v_j_bb)
    correlation_p_bb=sum([(i-m_v_i_bb)*(j-m_v_j_bb) for i,j in zip(v_i_bb,v_j_bb)])\
            /np.sqrt(sum([(i-m_v_i_bb)**2 for i in v_i_bb])*(sum([(j-m_v_j_bb)**2 for j in v_j_bb])))    
    return [correlation,correlation_p,correlation_p_bb]
def correlation_loop(L=None,iid_pair=None):
    '''
    pair
	'''
    #count=0    
    #from random import randint
    import random
    for_loop=random.sample(range(0,len(cells)),len(cells))
    #for_loop=[randint(0,len(cells)) for _ in range(int(len(cells)/2))]
    #print(for_loop)
    for j in for_loop:
        if len(iid_pair)>300:
            break
        if center[j][0]>0 and center[j][0]<1 and center[j][1]>0 \
            and center[j][1]<1:
            #only the central part
            point=Point(cells[j]['original'][0:2])
            construct_pair(cells,point,L=L,id_pair=iid_pair)
            #print(id_pair)

id_pair=[]
f=open('result','w')
#normalized by the minimum radius
for i in np.arange(2,9,0.4):
    print(i)
    correlation_loop(i*R_2,id_pair)
    #print(remove_repitition)
    #print(free_volume_norm)
    f.write(str(i)+' '+str(remove_repitition(id_pair,free_volume_norm))+'\n')
    id_pair=[] 
f.close()
plt.show()
'''
###############################plot radius
plt.figure()
h=sorted(volume_radius)
x=np.linspace(h[0],h[-1],1000)
weights=np.ones_like(h)/float(len(h))
plt.hist(h,int(num_lines/10),weights=weights,facecolor='blue',alpha=1)
plt.yscale('log',nonposy='clip')
fs=20
ts=18
ax=plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fs)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fs)
    tick.label1.set_fontweight('bold')
plt.rc('axes', linewidth=2)
plt.rc('axes',labelsize=fs)
plt.rc('xtick',labelsize=ts)
plt.rc('ytick',labelsize=ts)
plt.xlabel('Disk radius ',fontsize=fs,fontweight='bold')
plt.ylabel(r'$P(R)$',fontsize=fs+10,fontweight='bold')
plt.legend(loc='best',fontsize=fs)
'''