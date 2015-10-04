
# coding: utf-8

# In[14]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

#P1 Tarea 3

#edo a resolver d2x/dt2= -kx-u(x2-a2)dx/dt y con cambio de variable d2y/ds2=-y-u*(y2-1)dy/ds
def rk3(f,h,CT0,CI1,CI2): #metodo de runge-kutta orden 3 para edo orden 2 (ie dos edo's de orden 1)
    """
    -Esta funcion permite resolver una EDO de segundo orden
    -Recibe como argumento una funcion f, un paso de tiempo h, y condiciones iniciales para el tiempo (CT0), para la derivada de la solucion de la EDO (CT1) y para la solucion de la EDO (CT2)
    -Retorna el valor de la solucion a la EDO y su derivada en el tiempo CT0+h.
    """
    v0=CI1 #cond. inicial sobre derivada
    x0=CI2 #cond. inicial sobre posicion
    t0=CT0 #cond. inicial temporal
    k1=h*v0
    l1=h*f(x0,v0,t0)
    k2=h*(v0+l1/2.0)
    l2=h*f(x0+k1/2.0,v0+l1/2.0,t0+h/2.0)
    k3=h*(v0-l1+2.0*l2)
    l3=h*f(x0-k1+2.0*k2,v0-l1+2.0*l2,t0+h)
    xn=x0+(k1+4*k2+k3)/6.0
    vn=v0+(l1+4*l2+l3)/6.0
    return xn,vn #x_0+1,v_0+1=dx/dt_0+1
uast=1.808
#condiciones iniciales 1
v01=0
y01=0.1
#condiciones iniciales 2
v02=0
y02=4
h=0.1 #paso
y1=[y01] #solucion con condiciones iniciales 1
y2=[y02] #solucion con condiciones iniciales 2
v1=[v01] #dy/ds condiciones iniciales 1
v2=[v02] #dy/ds con condiciones iniciales 2
n=600 #aprox. 10 periodos (20 pi aprox. 60 => n*h=600*0.1=60)
for i in range(n):
    y01=(rk3(lambda y,v,t:-y-uast*(y**2-1)*v,h,0,v01,y01))[0] #con v=dy/ds 
    v01=(rk3(lambda y,v,t:-y-uast*(y**2-1)*v,h,0,v01,y01))[1]
    y1.append(y01)
    v1.append(v01)
for j in range(n):
    y02=(rk3(lambda y,v,t:-y-uast*(y**2-1)*v,h,0,v02,y02))[0]
    v02=(rk3(lambda y,v,t:-y-uast*(y**2-1)*v,h,0,v02,y02))[1]
    y2.append(y02)
    v2.append(v02)
s=np.linspace(0,n*h,num=n+1) #n+1 pq son n+1 valores debido a que la lista yi parte con el valor v0i
#grafico y vs s
fig1=plt.figure(1)
fig1.clf()
plt.title('Grafico y(s) vs s')
plt.xlabel('s')
plt.ylabel('y(s)')
plt.plot(s,y1,'r-')
plt.plot(s,y2,'b-')
red_patch = mpatches.Patch(color='red', label='Solucion con condicion inicial (1)')
blue_patch = mpatches.Patch(color='blue', label='Solucion con condicion inicial (2)')
plt.legend(handles=[red_patch,blue_patch])
fig1.savefig('yvss')
plt.grid(True)
#grafico y vs dy/ds
fig2=plt.figure(2)
fig2.clf()
plt.title('Grafico dy/ds vs y(s)')
plt.xlabel('y(s)')
plt.ylabel('dy/ds')
green_patch = mpatches.Patch(color='green', label='Derivada con condicion inicial (1)')
yellow_patch = mpatches.Patch(color='yellow', label='Derivada con condicion inicial (2)')
plt.legend(handles=[green_patch,yellow_patch])
plt.plot(y1,v1,'g-')
plt.plot(y2,v2,'y-')
plt.grid(True)
fig2.savefig('yvsdyds')
plt.show()

#P2 Tarea 3

#funcion para definir las ecs. de lorenz
def ecs_lorenz(t,w,p): #w es un vector de (x,y,z)
    """
    -Esta funcion recibe como argumentos un arreglo de tiempo t, un vector w=[x,y,z] y un arreglo de constantes p=[sigma,beta,rho]
    -Retorna las ecuaciones de lorenz para dicho vector.
    """
    sigma=p[0]
    beta=p[1]
    rho=p[2]
    dw = np.zeros([3])
    dw[0] = sigma*(w[1]-w[0]) #dx/ds
    dw[1] = w[0]*(rho - w[2])-w[1] #dy/ds
    dw[2] = w[0]*w[1]-beta*w[2] #dz/ds
    return dw
t0 = 0
tf = 100.0
dt = 0.01 #paso
w0 = [5,-5,10] #condiciones iniciales del vector w=(x,y,z) ie (x0,y0,z0)
Y=[]
T=[]
sol = ode(ecs_lorenz).set_integrator('vode',method='adams') #metodo de resolución de edo runge-kutta orden 4
p = [10.0,8.0/3.0,28.0]  #valores de sigma, beta y rho
sol.set_f_params(p).set_initial_value(w0,t0) #incluye parametros dentro de la solucion
while sol.successful() and sol.t+dt < tf:
    sol.integrate(sol.t+dt) #integra entre t y tf con dt como paso
    Y.append(sol.y)
    T.append(sol.t)
Y = np.array(Y) #cambia lista Y a un arreglo
fig3 = plt.figure(3)
fig3.clf()
ax = fig3.add_subplot(111, projection='3d')
ax.set_aspect('auto')
ax.set_title(r'$Grafico \ 3D \ de \ las \ soluciones \ a \ las \ ecs. \ de \ Lorenz$')
ax.plot(Y[:,0], Y[:,1],Y[:,2],'r-') #plot(x,y,z)
ax.set_xlabel(r'$Eje \ X$')
ax.set_ylabel(r'$Eje \ Y$')
ax.set_zlabel(r'$Eje \ Z$')
fig3.savefig('3d')
plt.show()




# In[ ]:



