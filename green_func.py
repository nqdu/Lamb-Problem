"""
Lamb's Problem, Second type.
Refer to Johnson(1974)： https://academic.oup.com/gji/article/37/1/99/678320

:copyright:
    Nanqiao Du (nqdu@foxmail.com), April 2019
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
import matplotlib.pyplot as plt 
from numpy import sqrt,pi,sin,cos
from numpy.polynomial.legendre import leggauss

def gaussint(func,a,b,param,args,nseg = 20):
	'''
	integrate over (a,b), with gauss-legendre quadrature

	Parameters:
		func : integrand function
		a,b  : integral interval
		param: gauss legendre nodes and weights for x in range (-1,1)
		args : additional parameters for func
		nseg : divide integral interval into nseg segments

	Returns:
		s : s=\\int_{a}^{b}func(x)dx
	'''
	x0,w=param
	n=len(x0)
	x=(x0*(b-a)+b+a)/2 
	s=0.0
	y=np.linspace(a,b,nseg)
	s = 0.0
	for i in range(len(y)-1):
		x=(x0*(y[i+1]-y[i])+y[i]+y[i+1])/2
		s=s+sum([func(x[_i],args)*w[_i] for _i in range(n)])*(y[i+1]-y[i])/2
	#s=sum([func(x[i],args)*w[i] for i in range(n)])*(b-a)/2

	return s

def greenmatrix(p,q,phi,a,b):
	'''
	Construct green matrix M,N for P and S wave, 
	and sigma,etaa,etab for calculation

	Parameters:
		p,q,theta,phi: see Johnson(1979)
		a,b : alpha,beta, P and S wave velocity

	Returns:
		M,N,sigma,etaa,etab
	'''
	etaa=(0.0j+1/a**2+p**2-q**2)**0.5
	etab=(0.0j+1/b**2+p**2-q**2)**0.5
	gamma=etab**2+p**2-q**2
	sigma=gamma**2+4*etab*etaa*(q**2-p**2)
	
	M=np.zeros((3,3),dtype = np.complex)
	M[0,0]=2*etab*((p**2+q**2)*cos(phi)**2-p**2)
	M[0,1]=2*etab*(p**2+q**2)*sin(phi)*cos(phi)
	M[0,2]=2*q*etab*etaa*cos(phi)
	M[1,0]=M[0,1]*1.0
	M[1,1]=2*etab*((p**2+q**2)*sin(phi)**2-p**2)
	M[1,2]=2*q*etaa*etab*sin(phi)
	M[2,0]=q*gamma*cos(phi)
	M[2,1]=q*gamma*sin(phi)
	M[2,2]=etaa*gamma
	
	N=np.zeros((3,3),dtype=np.complex)
	N[0,0]=1/etab*(etab**2*gamma-(gamma-4*etaa*etab)*((p**2+q**2)*sin(phi)**2-p**2))
	N[0,1]=1/etab*(q**2+p**2)*(gamma-4*etaa*etab)*sin(phi)*cos(phi)
	N[0,2]=-q*gamma*cos(phi)
	N[1,0]=N[0,1]*1.0
	N[1,1]=1/etab*(etab**2*gamma-(gamma-4*etaa*etab)*((p**2+q**2)*cos(phi)**2-p**2))
	N[1,2]=-q*gamma*sin(phi)
	N[2,0]=-2*q*etaa*etab*cos(phi)
	N[2,1]=-2*q*etaa*etab*sin(phi)
	N[2,2]=2*etaa*(q**2-p**2)
	
	return M,N,sigma,etaa,etab

def inta(p,args):
	'''
	Construct integrand for P wave, on the Cagniard path
	'''
	theta,phi,r,a,b,rho,t=args
	s=np.zeros((3,3))
	if t>r*sqrt(1/a**2+p**2):
		mu=rho*b**2
		q=-t/r*sin(theta)+1j*(sqrt((t/r)**2-1/a**2-p**2)*cos(theta))
		M,_,sigma,etaa,_=greenmatrix(p,q,phi,a,b)
		sp=etaa/sigma/sqrt((t/r)**2-1/a**2-p**2)*M
		s=sp.real/(pi**2*mu*r)
	return s

def intb(p,args):
	'''
	Construct integrand for S wave, on the second part of Cagniard path
	'''
	theta,phi,r,a,b,rho,t=args
	s=np.zeros((3,3))
	if t>r*sqrt(1/b**2+p**2):
		mu=rho*b**2
		q=-t/r*sin(theta)+1j*(sqrt((t/r)**2-1/b**2-p**2)*cos(theta))
		_,N,sigma,_,etab=greenmatrix(p,q,phi,a,b)
		sp=etab/sigma/sqrt((t/r)**2-1/b**2-p**2)*N
		s=sp.real/(pi**2*mu*r)
	return s

def intb2(p,args):
	'''
	Construct integrand for S wave, on the branch cut

	'''
	theta,phi,r,a,b,rho,t=args
	s=np.zeros((3,3))
	if t<r*sqrt(1/b**2+p**2):
		mu=rho*b**2
		q=-t/r*sin(theta)+sqrt(-(t/r)**2+1/b**2+p**2)*cos(theta)
		_,N,sigma,_,etab=greenmatrix(p,q,phi,a,b)
		sp=etab/sigma/sqrt(-(t/r)**2+1/b**2+p**2)*N
		s=sp.imag/(pi**2*mu*r)
		return s

def green(theta,phi,r,a,b,rho,t,gauss_points):
	'''
	Compute g^H in time domain

	Parameters:
		theta,phi,r : source and receiver parameters
		a,b,rho  : media parameters
		t : time range

	Returns:
		G : green's function, convolved with a Heaviside funciton,3*3 matrix
	'''
	nt=t.shape[0]

	# allocate space
	P=np.zeros((nt,3,3))
	S=np.zeros((nt,3,3))

	# Generate gauss-legendre points
	param=leggauss(gauss_points)

	# compute green's function
	t2=r/a*sin(theta)+r*cos(theta)*sqrt(1/b**2-1/a**2)
	for i in range(nt):
		t0 = t[i]
		print("computing green's function:  %f%% "%((i + 1) / nt * 100))
		pp=(t0/r)**2-(1/a)**2
		pb=(t0/r)**2-(1/b)**2
		p2=((t0/r-sqrt(1/b**2-1/a**2)*cos(theta))/sin(theta))**2-1/a**2
		args=(theta,phi,r,a,b,rho,t0)
		
		if pp>0:
			P[i,:,:]=gaussint(inta,0,sqrt(pp),param,args)
		if pb>0:
			S[i,:,:]=gaussint(intb,0,sqrt(pb),param,args)
		if sin(theta)>b/a:
			if t0<r/b and t0>t2:
				S[i,:,:] -= gaussint(intb2,0,sqrt(p2),param,args)
			elif t0>r/b:
				S[i,:,:] -= gaussint(intb2,sqrt(pb),sqrt(p2),param,args)
		
	return P+S

def show_green(x,xs,r,a,b,theta,t,G):
	'''
	Show green's function g^H, only 11,13,22,31,33 components
	'''
	plt.figure(figsize=(6,6))
	left=[0,2,1,0,2]
	right=[0,0,1,2,2]
	for i in range(5):
		g=G[:,left[i],right[i]]
		if i!=4:
			plt.plot(t,g*1000+(5-i)*0.2,'k')
		else:
			plt.plot(t,g*1000+(5-i)*0.0,'k')

		strtext='$g_{'+str(left[i]+1)+str(right[i]+1)+'}^{H}$'
		strtitle='$\\mathbf{G}^{H}('
		for j in range(3):
			strtitle+=str(x[j])+','
		strtitle+='t;'
		for j in range(3):
			strtitle+=str(xs[j])+','
		strtitle+='0'+')$'
		if i!=4:
			plt.text(0.65,(5-i)*0.2,strtext,fontsize=16)
		else:
			plt.text(0.65,(5-i)*0.0,strtext,fontsize=16)

	plt.title(strtitle)
	plt.xlabel('t/s')
	plt.yticks([])
	plt.xlim(0.5,4)
	plt.show()

def main(): 
	# you could change parameters here
	#=======================================================
	# media parameters, here we don't convert it to SI units
	x=[10.0,0,0] # receiver
	xs=[0,0,2.0] # source , make sure xs[2] > 0
	a=8.0 # vp
	b=4.62 # vs
	rho=3.3 # density

	# time vector
	dt=0.01 # time interval
	t=np.arange(1,4+dt,dt)

 	# if xs[2] is small, please increase this number,
	# or increase "nseg" in function `gaussint`
	gauss_points=7

	# end parameters
	# ============================================================

	# Utilize that the y component of source and field point are
	# both 0, this simplify the calculation of theta and phi
	#theta=np.arctan(x[0]/xs[2])
	#phi=np.arctan2(xs[1],x[0])
	r=sqrt(sum([(x[i]-xs[i])**2 for i in range(3)]))
	r_plane = sqrt(sum([(x[i]-xs[i])**2 for i in range(2)]))
	theta = np.arctan(r_plane / (-x[2] + xs[2]))
	phi = np.arctan2(x[1] - xs[1],x[0] - xs[0])

	# Compute and show g
	G=green(theta,phi,r,a,b,rho,t,gauss_points)
	show_green(x,xs,r,a,b,theta,t,G)

if __name__ == '__main__':
	main()
