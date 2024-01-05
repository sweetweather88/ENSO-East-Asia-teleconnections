import sys
import numpy as np
import datetime as datetime
from netCDF4 import Dataset
from scipy import interpolate
from  windspharm.standard import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim


######################################################################################
# Theory : Karoly 1983                                                               #
# Original Matlab code: Written by Jeff Shaman + Some minor changes by Eli Tziperman #
# Matlab code --> python : Irina Rudeva                                              #
# Modified by Ye-Jun Jun                                                             #
######################################################################################


#####################
#-----PARAMETER-----#
#####################
pi=np.pi
dtr=pi/180
rtd=180/pi
radius=6.371e6
e_omega=7.292e-5
day2s=24*60*60
min2s=60


###################
#-----SETTING-----#
###################
#Main
wavenumbers=np.array([1,2,3,4,5])
lon0=45; lat0=30
syear=1979; eyear=2021
level=200

#Others
Periods=np.array([float('inf'),15])*day2s
freq=2*pi/Periods
nfreq=freq.shape[0]
complex_tracing=False
qdifferent=1
#dt = 15 * min2s   #time increment in s, dt = 15 min
#intg_time=30*day2s  #integration time
dt=30*min2s
intg_time=15*day2s
Nsteps=int(intg_time/dt)

mm=np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
bgs='DJF'
if bgs == 'DJF':
	bgmon=np.array([12,1,2])


#####################
#-----READ DATA-----#
#####################
#For dimension
idir="/eddy_data2/REANA/ERA5/monthly/"
fname="/U/1.5x1.5/U.197901.nc"
print("Read Data for dimension: "+idir+fname)

ncu0=Dataset(idir+fname,'r')
dimnam0=('lon','lat','lev','time')
varnam0=['lon','lat','lev','time','U']

lons=ncu0.variables[varnam0[0]][:]
lats=ncu0.variables[varnam0[1]][:]
levs=ncu0.variables[varnam0[2]][:]
time=ncu0.variables[varnam0[3]][:]

iz0=np.where(levs == level)
iz=iz0[0][0]

uwnd0=ncu0.variables['U'][0,iz,:,:]
uwnds=np.zeros_like(uwnd0)
vwnds=np.zeros_like(uwnd0)

#For calculation
nmon=0
for yr in range(syear,eyear):
	for m in bgmon:
		fu=idir+"/U/1.5x1.5/U."+str(yr)+mm[m-1]+".nc"
		fv=idir+"/V/1.5x1.5/V."+str(yr)+mm[m-1]+".nc"
		ncu=Dataset(fu,'r')
		ncv=Dataset(fv,'r')
		if nmon == 0:
			uwnds=ncu.variables["U"][0,iz,:,:]
			vwnds=ncv.variables["V"][0,iz,:,:]

		elif nmon > 0:
			uwnds=uwnds+ncu.variables["U"][0,iz,:,:]
			vwnds=vwnds+ncv.variables["V"][0,iz,:,:]

		nmon += 1
nmon += 1
u=uwnds/nmon
v=vwnds/nmon


##############################
#-----BASIC CALCU30IONS-----#
##############################
#Streamfunction & absolute vorticity
w=VectorWind(u,v)
psi=w.streamfunction()
q0=w.absolutevorticity()

xm=lons*radius*dtr
xm360=360*radius*dtr
coslat=np.cos(dtr*lats)
cos2=np.power(coslat,2)
wherezero=float('inf')
um=u/coslat[:,None]
vm=v/coslat[:,None]
ym=np.zeros_like(lats)

for i in range(len(lats)):  
	if np.sin(dtr*lats[i]) == 0:
		ym[i]=radius*np.log((1+np.sin(1e-10))/np.cos(dtr*lats[i]))
	else:
		ym[i]=radius*np.log((1+np.sin(dtr*lats[i]))/np.cos(dtr*lats[i]))

dx=np.gradient(xm)
dy=np.gradient(ym)

psix=np.zeros_like(psi) 
psixx=np.zeros_like(psi) 
psiy=np.zeros_like(psi) 
psiyy=np.zeros_like(psi)
q=np.zeros_like(psi)

for i in range(np.shape(psi)[0]):  
	psix[i,:]=np.gradient(psi[i,:])/dx[:]  

for i in range(np.shape(psi)[0]):  
	psixx[i,:]=np.gradient(psix[i,:])/dx[:]  

for j in range(np.shape(psi)[1]):  
	for i in range(1,np.shape(psi)[0]-1):
		psiy[i,j]=(psi[i+1,j]-psi[i-1,j])/(ym[i+1]-ym[i-1])
		psiy[np.shape(psi)[0]-1,j]=(psi[np.shape(psi)[0]-1,j]-psi[np.shape(psi)[0]-2,j])/(ym[np.shape(psi)[0]-1]-ym[np.shape(psi)[0]-2])
		psiy[0,j]=(psi[0,j]-psi[1,j])/(ym[0]-ym[1])

for j in range(np.shape(psi)[1]):  
	for i in range(1,np.shape(psi)[0]-1):
		psiyy[i,j]=(psiy[i+1,j]-psiy[i-1,j])/(ym[i+1]-ym[i-1])
		psiyy[np.shape(psi)[0]-1,j]=(psiy[np.shape(psi)[0]-1,j]-psiy[np.shape(psi)[0]-2,j])/(ym[np.shape(psi)[0]-1]-ym[np.shape(psi)[0]-2])
		psiyy[0,j]=(psiy[0,j]-psiy[1,j])/(ym[0]-ym[1])

for j in range(np.shape(psi)[1]):  
	q[:,j]=(psixx[:,j]+psiyy[:,j])/cos2+2*e_omega*np.sin(dtr*lats)

#BetaM in zonally varing bg
cosuy=np.zeros_like(um)
cosuyy=np.zeros_like(um)
BetaM=np.zeros_like(um)
umx=np.zeros_like(um)
vmx=np.zeros_like(vm)
qx=np.zeros_like(q)
qxx=np.zeros_like(q)
qxy=np.zeros_like(q)
umy=np.zeros_like(um)
vmy= np.zeros_like(vm)
qy=np.zeros_like(q)
qyy=np.zeros_like(q)
tmp=2*e_omega*cos2/radius

for j in range(np.shape(um)[1]):
	for i in range(1,np.shape(um)[0]-1):
		cosuy[i,j]=(um[i+1,j]*cos2[i+1]-um[i-1,j]*cos2[i-1])/(ym[i+1]-ym[i-1])
	cosuy[np.shape(psi)[0]-1,j]=(um[np.shape(psi)[0]-1,j]*cos2[np.shape(psi)[0]-1]-um[np.shape(psi)[0]-2,j]*cos2[np.shape(psi)[0]-2])/(ym[np.shape(psi)[0]-1]-ym[np.shape(psi)[0]-2])
	cosuy[0,j]=(um[0,j]*cos2[0]-um[1,j]*cos2[1])/(ym[0]-ym[1])

for j in range(np.shape(um)[1]):
	for i in range(1,np.shape(um)[0]-1):
		cosuyy[i,j]=(cosuy[i+1,j]/cos2[i+1]-cosuy[i-1,j]/cos2[i-1])/(ym[i+1]-ym[i-1])
	cosuyy[np.shape(psi)[0]-1,j]=(cosuy[np.shape(psi)[0]-1,j]/cos2[np.shape(psi)[0]-1]-cosuy[np.shape(psi)[0]-2,j]/cos2[np.shape(psi)[0]-2])/(ym[np.shape(psi)[0]-1]-ym[np.shape(psi)[0]-2])
	cosuyy[0,j]=(cosuy[0,j]/cos2[0]-cosuy[1,j]/cos2[1])/(ym[0]-ym[1])

for j in range(np.shape(um)[1]):
	BetaM[:,j]=tmp[:]-cosuyy[:,j]

for i in range(np.shape(um)[0]):
	umx[i,:]=np.gradient(um[i,:])/dx[:]
	vmx[i,:]=np.gradient(vm[i,:])/dx[:]
	qx[i,:]=np.gradient(q[i,:])/dx[:]
	qxx[i,:]=np.gradient(qx[i,:])/dx[:]

for j in range(np.shape(psi)[1]):  
	for i in range(1,np.shape(psi)[0]-1):
		umy[i,j]=(um[i+1,j]-um[i-1,j])/(ym[i+1]-ym[i-1])
		vmy[i,j]=(vm[i+1,j]-vm[i-1,j])/(ym[i+1]-ym[i-1])
		qy[i,j]=(q[i+1,j]-q[i-1,j])/(ym[i+1]-ym[i-1])
		qyy[i,j]=(qy[i+1,j]-qy[i-1,j])/(ym[i+1]-ym[i-1])
		qxy[i,j]=(qx[i+1,j]-qx[i-1,j])/(ym[i+1]-ym[i-1])

	umy[np.shape(psi)[0]-1,j]=(um[np.shape(psi)[0]-1,j]-um[np.shape(psi)[0]-2,j])/(ym[np.shape(psi)[0]-1]-ym[np.shape(psi)[0]-2])
	vmy[np.shape(psi)[0]-1,j]=(vm[np.shape(psi)[0]-1,j]-vm[np.shape(psi)[0]-2,j])/(ym[np.shape(psi)[0]-1]-ym[np.shape(psi)[0]-2])
	qy[np.shape(psi)[0]-1,j]=(q[np.shape(psi)[0]-1,j]-q[np.shape(psi)[0]-2,j])/(ym[np.shape(psi)[0]-1]-ym[np.shape(psi)[0]-2])
	qyy[np.shape(psi)[0]-1,j]=(qy[np.shape(psi)[0]-1,j]-qy[np.shape(psi)[0]-2,j])/(ym[np.shape(psi)[0]-1]-ym[np.shape(psi)[0]-2])
	qxy[np.shape(psi)[0]-1,j]=(qx[np.shape(psi)[0]-1,j]-qx[np.shape(psi)[0]-2,j])/(ym[np.shape(psi)[0]-1]-ym[np.shape(psi)[0]-2])

	umy[0,j]=(um[0,j]-um[1,j])/(ym[0]-ym[1])
	vmy[0,j]=(vm[0,j]-vm[1,j])/(ym[0]-ym[1])
	qy[0,j]=(q[0,j]-q[1,j])/(ym[0]-ym[1])
	qyy[0,j]=(qy[0,j]-qy[1,j])/(ym[0]-ym[1])
	qxy[0,j]=(qx[0,j]-qx[1,j])/(ym[0]-ym[1])


#####################
#-----FUNCTIONS-----#
#####################
#Basic
def Kt(k,um,fr,BetaM):
	Kt = np.nan
	if BetaM > 0 and um-fr/k > 0:
		Kt = np.sqrt(BetaM/(um-fr/k))
	return Kt

def ug(k,l,um,qx,qy) :
	Ks2=k**2+l**2
	Ks4=Ks2**2
	return um+((k**2-l**2)*qy-2*k*l*qx)/Ks4

def vg(k,l,vm,qx,qy) :
	Ks2=k**2+l**2
	Ks4=Ks2**2
	return vm+((k**2-l**2)*qx+2*k*l*qy)/Ks4

def kt(k,l,umx,vmx,qxy,qxx) :
	Ks2=k**2+l**2
	return -k*umx-l*vmx+(qxy*k-qxx*l)/Ks2

def lt(k,l,umy,vmy,qxy,qyy) :
	Ks2=k**2+l**2
	return -k*umy-l*vmy+(qyy*k-qxy*l)/Ks2

def y2lat(a):
	return a/(dtr*radius)

def lat2y(a):
	return a*dtr*radius

#Numerical
def rk(x,y,k,l):
	xt=ug(k,l,umint(x,y),qxint(x,y),qyint(x,y))
	yt=vg(k,l,vmint(x,y),qxint(x,y),qyint(x,y))
	dkdt=kt(k,l,umxint(x,y),vmxint(x,y),qxyint(x,y),qxxint(x,y))
	dldt=lt(k,l,umyint(x,y),vmyint(x,y),qxyint(x,y),qyyint(x,y))
	return xt,yt,dkdt,dldt

def RK(x1,y1,k1,l1,dt):
	kx1,ky1,kk1,kl1=rk(x1,y1,k1,l1)

	if kl1 != -1:
		x2=x1+kx1*dt/2
		y2=y1+ky1*dt/2
		k2=k1+kk1*dt/2
		l2=l1+kl1*dt/2
		kx2,ky2,kk2,kl2=rk(x2,y2,k2,l2)

		if kl2 != -1:
			x3=x1+kx2*dt/2
			y3=y1+ky2*dt/2
			k3=k1+kk2*dt/2
			l3=l1+kl2*dt/2
			kx3,ky3,kk3,kl3=rk(x3,y3,k3,l3)

			if kl3 != -1:
				x4=x1+kx3*dt
				y4=y1+ky3*dt
				k4=k1+kk3*dt
				l4=l1+kl3*dt
				kx4,ky4,kk4,kl4=rk(x4,y4,k4,l4)

				if kl4 != -1:
					dx=dt*(kx1+2*kx2+2*kx3+kx4)/6
					dy=dt*(ky1+2*ky2+2*ky3+ky4)/6
					dk=dt*(kk1+2*kk2+2*kk3+kk4)/6
					dl=dt*(kl1+2*kl2+2*kl3+kl4)/6
					return dx,dy,dk,dl


########################
#-----WRITE NETCDF-----#
########################
ymout=np.zeros_like(q)
ymout[:,0]=ym
varlist=np.zeros(18,dtype={'names':['name','outname','data','scale'],'formats':["a5","a5",'(121,240)f4','f4']})

varlist[0]=("u","u",u,1)
varlist[1]=("v","v",v,1)
varlist[2]=("um","um",um,1)
varlist[3]=("vm","vm",vm,1)
varlist[4]=("umx","umx",umx,1e-6)
varlist[5]=("umy","umy",umy,1e-6)
varlist[6]=("vmx","vmx",vmx,1e-6)
varlist[7]=("vmy","vmy",vmy,1e-6)
varlist[8]=("qbar","q",q,1e-4)
varlist[9]=("qx","qx",qx,1e-12)
varlist[10]=("qy","qy",qy,1e-11)
varlist[11]=("qxx","qxx",qxx,1e-18)
varlist[12]=("qyy","qyy",qyy,1e-18)
varlist[13]=("qxy","qxy",qxy,1e-18)
varlist[14]=("BetaM","BetaM",BetaM,1e-11)
varlist[15]=("sf","psi",psi,1e+8)
varlist[16]=("qbar0","q0",q0,1e-4)
varlist[17]=("ym","ym",ymout,1)

for iv in range(len(varlist)):
	ncvar=varlist["outname"][iv]
	ftest='../../output_final/io/2/test.%s.nc'%(varlist["outname"][iv])
	ncout=Dataset(ftest,'w',format='NETCDF4')
	ncout.description="TEST %s"%(ftest)

	dimnam=('lon','lat','time')
	varnam=['lon','lat','time',ncvar]

	ncout.createDimension(dimnam[0],len(lons))
	ncout.createDimension(dimnam[1],len(lats))

	for nv in range(2):
		ncout_var=ncout.createVariable(varnam[nv],ncu0.variables[varnam[nv]].dtype,dimnam[nv])
		for ncattr in ncu0.variables[varnam[nv]].ncattrs():
			ncout_var.setncattr(ncattr,ncu0.variables[varnam[nv]].getncattr(ncattr))

	ncout.variables[dimnam[0]][:]=lons
	ncout.variables[dimnam[1]][:]=lats

	ncout_var=ncout.createVariable(ncvar,'f',dimnam[1::-1])
	ncout_var.scale_factor=varlist["scale"][iv]
	ncout_var.add_offset=0
	ncout_var.units='scale   %s'%varlist["scale"][iv]
	ncout_var[:]=varlist["data"][iv]
	ncout.close()

ncu0.close()
ncu.close()
ncv.close()
print("All derivatives are done!")


#########################
#-----INTERPO30ION-----#
#########################
print("")
print("Interpolation")

uint=interpolate.interp2d(xm,ym[1:-1],u[1:-1,:],kind='cubic')
vint=interpolate.interp2d(xm,ym[1:-1],v[1:-1,:],kind='cubic')
umint=interpolate.interp2d(xm,ym[1:-1],um[1:-1,:],kind='cubic')
vmint=interpolate.interp2d(xm,ym[1:-1],vm[1:-1,:],kind='cubic')
umxint=interpolate.interp2d(xm,ym[1:-1],umx[1:-1,:],kind='cubic')
umyint=interpolate.interp2d(xm,ym[1:-1],umy[1:-1,:],kind='cubic')
vmxint=interpolate.interp2d(xm,ym[1:-1],vmx[1:-1,:],kind='cubic')
vmyint=interpolate.interp2d(xm,ym[1:-1],vmy[1:-1,:],kind='cubic')
qint=interpolate.interp2d(xm,ym[1:-1],q[1:-1,:],kind='cubic')
qxint=interpolate.interp2d(xm,ym[1:-1],qx[1:-1,:],kind='cubic')
qyint=interpolate.interp2d(xm,ym[1:-1],qy[1:-1,:],kind='cubic')
qxxint=interpolate.interp2d(xm,ym[1:-1],qxx[1:-1,:],kind='cubic')
qyyint=interpolate.interp2d(xm,ym[1:-1],qyy[1:-1,:],kind='cubic')
qxyint=interpolate.interp2d(xm,ym[1:-1],qxy[1:-1,:],kind='cubic')
BetaMint=interpolate.interp2d(xm,ym[1:-1],BetaM[1:-1,:],kind='cubic')

#Ray tracing
j=np.argmin(np.absolute(lons-lon0))
i=np.argmin(np.absolute(lats-lat0))

ifr=-1
for fr in freq:
	ifr=ifr+1
	print("  Ray tracing: period", Periods[ifr])

	for k in wavenumbers:
		print("")
		print("  initial k = ",k)

		spotk=k/(radius*coslat[i])
		print("spotk=", spotk)
		coeff = np.zeros(4)
		coeff[0]=vm[i,j]
		coeff[1]=um[i,j]*spotk-fr;
		coeff[2]=vm[i,j]*spotk*spotk+qx[i,j]
		coeff[3]=um[i,j]*np.power(spotk,3)-qy[i,j]*spotk-fr*spotk*spotk

		lroot = np.roots(coeff)
		print("  initial l = ", lroot*radius)

		for R in range(0,3):
			spotl=lroot[R]
			print("  Root # ", R, "  spotl = ", spotl, spotl*radius)

			if complex_tracing is False :
				if np.not_equal(np.imag(spotl),0) :
					print("   *** found complex initial l, not tracing. ")
					print("   *** Ray tracing: period", Periods[ifr])
					print("   *** initial k ", k)
					print("   *** Root # \n", R)
					continue

			lonn=np.empty(Nsteps+1)
			latn=np.empty(Nsteps+1)
			xn=np.empty(Nsteps+1)
			yn=np.empty(Nsteps+1)
			kn=np.empty(Nsteps+1)
			ln=np.empty(Nsteps+1)

			lonn[:]=np.nan
			latn[:]=np.nan
			xn[:]=np.nan
			yn[:]=np.nan
			kn[:]=np.nan
			ln[:]=np.nan

			for t in range(0,Nsteps):
				if np.equal(np.remainder(t,40),0):
					print("   t = ",t)

				if t==0:
					x0=xn[0]=xm[j]
					y0=yn[0]=ym[i]
					k0=kn[0]=spotk
					l0=ln[0]=np.real(spotl)
					lonn[0]=lon0
					latn[0]=lat0

				else:
					x0=xn[t]
					y0=yn[t]
					k0=kn[t]
					l0=np.square(Kt(k0,umint(x0,y0),fr,BetaMint(x0,y0)))-k0*k0

					if l0 >= 0:
						l0=np.sqrt(l0)
						if ln[t] >= 0:
							l0=l0
						else:
							l0=-l0

					else:
						l0=np.nan
					l0=ln[t]

				if np.isnan(l0):
					break

				dx,dy,dk,dl=RK(x0,y0,k0,l0,dt)

				tn=t+1
				xn[tn]=x0+dx
				if xn[tn] >= xm360:
					xn[tn]=xn[tn]-xm360
				yn[tn]=y0+dy
				kn[tn]=k0+dk
				ln[tn]=l0+dl

				if np.isnan(dl):
					print('Ray terminated: vg =',dy/dt)
					break

				lonn[tn]=xn[tn]*rtd/radius
				latn[tn]=y2lat(yn[tn])

			if fr==0:
				fout=open('../../output_final/io/2/raypath_zv_{:s}_loc{:d}N_{:d}E_period{}_k{:d}_root{:d}'.format(bgs,lat0,lon0,'_inf',k,R),'w')

			else:
				fout=open('../../output_final/io/2/raypath_zv_{:s}_loc{:d}N_{:d}E_period{:0.0f}_k{:d}_root{:d}'.format(bgs,lat0,lon0,2*pi/(fr*day2s),k,R),'w')

			frmt="{:>5} {:>3} {:>4}"+" {:>6}"*2+(" {:>6}"+" {:>9}")*3+" {:>9}"*3+" {:>7}"*4+" {:>9}"*11+" \n"
			fout.write(frmt.format('t','hr','day','lon','lat','k*rad','k','l*rad','l','l0*rad','l0','K','KK','Kom','u','v','um','vm','umx','vmx','umy','vmy','q','qx','qy','BetaM','qxx','qyy','qxy'))
			frmt="{:>5} {:>3} {:>4.1f}"+" {:>6.2f}"*2+(" {:>6.2f}"+" {:>9.2e}")*3+" {:>9.2f}"+" {:>9.2e}"*2+" {:>7.2f}"*4+" {:>9.2e}"*11+" \n"

			for t in range(0,Nsteps+1,int(12*3600/dt)):
				x=xn[t]
				y=yn[t]

				KK=np.sqrt(kn[t]*kn[t]*np.square(np.cos(latn[t]*dtr))+ln[t]*ln[t])*radius
				KK1=np.sqrt(kn[t]*kn[t]+ln[t]*ln[t])
				KKom=Kt(kn[t],umint(x,y),fr,BetaMint(x,y))

				if np.isnan(KKom):
					KKom=np.array([np.nan])

				l0=np.square(Kt(kn[t],umint(x,y),fr,BetaMint(x,y)))-kn[t]*kn[t]

				if l0 >= 0:
					l0=np.sqrt(l0)
					if ln[t] >= 0:
						l0=l0
					else:
						l0=-l0
				else:
					l0=np.array([np.nan])
				fout.write(frmt.format(t,t*dt/3600,t*dt/(3600*24.),lonn[t],latn[t],kn[t]*radius*np.cos(latn[t]*dtr),kn[t],ln[t]*radius,ln[t],l0[0]*radius,l0[0],KK,KK1,KKom[0],uint(x,y)[0],vint(x,y)[0],umint(x,y)[0],vmint(x,y)[0],umxint(x,y)[0],vmxint(x,y)[0],umyint(x,y)[0],vmyint(x,y)[0],qint(x,y)[0],qxint(x,y)[0],qyint(x,y)[0],BetaMint(x,y)[0],qxxint(x,y)[0],qyyint(x,y)[0],qxyint(x,y)[0]))
			fout.close()


	print("Ray tracing is finished!!!")
