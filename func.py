import numpy as np
import scipy.signal as signal
import scipy.constants as const
import matplotlib.pyplot as plt

__name__='func'

def array_geo(radius,sep,shape='cir'):
    if shape=='cir':
        r_ant=radius*np.ones(360//sep)
        theta_ant=np.radians(np.arange(0,360,sep))
    return np.array([r_ant,theta_ant])

def source_signal(Fsrc,Flo,Fsamp,Nsamp,As=np.array([1e-6,1e-6])):

    time=np.arange(0,Nsamp,1)*(1/Fsamp)
    As=np.expand_dims(As,axis=1)
    As=np.repeat(As,time.size,axis=1)
    s_src=As*np.exp(-1j*2*np.pi*np.outer((Fsrc-Flo),time))

    return s_src

def rx_signal(r_ant,theta_ant,rsrc,thetasrc,s_src,Fsrc,alpha=-50):

    wlsrc=3e8/Fsrc
    wlsrc=np.expand_dims(wlsrc,axis=1)
    wlsrc=np.repeat(wlsrc,r_ant.size,axis=1)

    rsrc=np.expand_dims(rsrc,axis=1)
    thetasrc=np.expand_dims(thetasrc,axis=1)
    rsrc=np.repeat(rsrc,r_ant.size,axis=1)
    thetasrc=np.repeat(thetasrc,theta_ant.size,axis=1)

    r_ant=np.expand_dims(r_ant,axis=0)
    theta_ant=np.expand_dims(theta_ant,axis=0)
    r_ant=np.repeat(r_ant,rsrc.shape[0],axis=0)
    theta_ant=np.repeat(theta_ant,thetasrc.shape[0],axis=0)

    z=np.sqrt(rsrc**2+r_ant**2-2*rsrc*r_ant*np.cos(theta_ant-thetasrc))
    a=(1/(z))*np.exp(-1j*2*np.pi*z/wlsrc)
    noise_var=alpha

    wgn=np.random.multivariate_normal(np.zeros(2),0.5*np.eye(2)*np.sqrt(noise_var),size=(r_ant.shape[1],s_src.shape[1])).view(np.complex128)
    wgn=wgn.reshape(r_ant.shape[1],s_src.shape[1])
    #print(wgn.shape)
    #print(np.matmul(a.T,s_src).shape)
    x=np.matmul(a.T,s_src)+wgn
    #for m in range(r_ant.size):
    #    x[m]=a[m]*s_src+np.random.normal(loc=0,scale=np.sqrt(noise_var),size=s_src.size)

    return x

def fft(x):
    return np.fft.fft(x)

def ifft(X):
    return np.fft.ifft(X)

def covariance_mat(x,per_snapshot=False):
    Nch=x[:,0].size
    Nscrs=1
    Nsamp=x[0,:].size
    if not per_snapshot:
        R=(1/Nch)*np.matmul(x,x.conjugate().T)
        evals,evecs=np.linalg.eig(R)

        noise=evecs[:,Nscrs:]
    elif per_snapshot:
        noise=np.zeros((Nch,Nch-1,Nsamp),dtype=np.complex128)
        for n in range(Nsamp):
            x_snap=x[:,n].reshape((Nch,1))
            R=(1/Nch)*np.matmul(x_snap,x_snap.conjugate().T)
            evals,evecs=np.linalg.eig(R)

            noise[:,:,n]=evecs[:,Nscrs:]

    return R,evals,evecs,noise

def position_estimation(U,gridsize,Ngrid,r_ant,theta_ant,Flo,grid='cart',timedep=False,omega=0,Nsamp=8192,Fsamp=200e6):
    if grid in ['cart','pol']:
        if grid=='cart':
            ind=np.arange(-(np.sqrt(Ngrid)+1)//2+1,(np.sqrt(Ngrid)+1)//2,1)
            x=y=2*gridsize*ind/np.sqrt(Ngrid)
            xx,yy=np.meshgrid(x,y,indexing='xy',sparse=False)
            R0=np.sqrt(xx**2+yy**2)
            T0=np.arctan2(yy,xx)
        elif grid=='pol':
            ind=np.arange(0,np.sqrt(Ngrid),1)
            r=gridsize*ind/np.sqrt(Ngrid)
            t=2*np.pi*ind/np.sqrt(Ngrid)
            R0,T0=np.meshgrid(r,t,indexing='xy',sparse=False)

        R0=np.expand_dims(R0,axis=0)
        T0=np.expand_dims(T0,axis=0)
        #print(T)
        R0=np.repeat(R0,r_ant.size,axis=0)
        T0=np.repeat(T0,r_ant.size,axis=0)
        r_ant=np.expand_dims(r_ant,axis=(1,2))
        r_ant=np.repeat(r_ant,R0.shape[1],axis=(1))
        r_ant=np.repeat(r_ant,R0.shape[1],axis=(2))
        theta_ant=np.expand_dims(theta_ant,axis=(1,2))
        theta_ant=np.repeat(theta_ant,T0.shape[1],axis=(1))
        theta_ant=np.repeat(theta_ant,T0.shape[1],axis=(2))

    elif grid =='fib':
        ind=np.arange(0,Ngrid,1)
        ep=1/2
        R0=gridsize*((ind+ep)/(Ngrid-1+2*ep))**.5
        T0=2*np.pi*ind*const.golden

        R0=np.expand_dims(R0,axis=0)
        R0=np.repeat(R0,r_ant.size,axis=0)
        T0=np.expand_dims(T0,axis=0)
        T0=np.repeat(T0,theta_ant.size,axis=0)
        #print(R.shape)

        #print(r.shape,theta.shape,r_ant.shape)
        r_ant=np.expand_dims(r_ant,axis=(1))
        r_ant=np.repeat(r_ant,R0.shape[1],axis=(1))

        theta_ant=np.expand_dims(theta_ant,axis=(1))
        theta_ant=np.repeat(theta_ant,T0.shape[1],axis=(1))
        #print(r_ant.shape)

    wl=3e8/Flo
    if not timedep:
        R=R0
        T=T0
        z=np.sqrt(R**2+r_ant**2-2*r_ant*R*np.cos(T-theta_ant))
        a=(1/z)*np.exp(-1j*2*np.pi*z/wl)

        p=abs(np.matmul(a.conjugate().T,U))**2
        if grid=='fib':
            P=1/np.sum(p,axis=1)
        else:
            P=1/np.sum(p,axis=2)

        if grid=='fib':
            return P,R[0,:],T[0,:]
        else:
            return P,R,T

    elif timedep:
        if grid=='fib':
            R=np.expand_dims(R0,axis=2)
            R=np.repeat(R,Nsamp,axis=2)
            T=np.expand_dims(T0,axis=2)
            T=np.repeat(T,Nsamp,axis=2)

            time=np.arange(0,Nsamp,1)*1/Fsamp
            omegatime=omega*time
            omegatime=np.expand_dims(omegatime,axis=0)
            omegatime=np.expand_dims(omegatime,axis=0)
            omegatime=np.repeat(omegatime,R.shape[0],axis=0)
            omegatime=np.repeat(omegatime,R.shape[1],axis=1)

            R=R
            T=T+omegatime

            r_ant=np.expand_dims(r_ant,axis=2)
            r_ant=np.repeat(r_ant,Nsamp,axis=2)
            theta_ant=np.expand_dims(theta_ant,axis=2)
            theta_ant=np.repeat(theta_ant,Nsamp,axis=2)

            z=np.sqrt(R**2+r_ant**2-2*r_ant*R*np.cos(T-theta_ant))
            a=(1/z)*np.exp(-1j*2*np.pi*z/wl)
            P=np.zeros((Ngrid,Nsamp))
            for n in range(Nsamp):

                p=abs(np.matmul(a[:,:,n].conjugate().T,U[:,:,n]))**2
                p=p.sum(1)
                P[:,n]=1/p

            return P,R[0,:],T[0,:]


def window_data(X,windowcenter,windowlen,window='blackmanharris'):
    N=X[0,:].size
    window=signal.windows.get_window(window,windowlen,fftbins=False)

    padded_window=np.concatenate((np.zeros(windowcenter-windowlen//2-1),window,np.zeros(X[0].size-windowcenter-windowlen//2)))
    padded_window=np.expand_dims(padded_window,axis=0)
    padded_window=np.repeat(padded_window,X[:,0].size,axis=0)
    filtered_signals=X*padded_window
    #print(X.shape,padded_window.shape)


    return filtered_signals

def antispiral_shift(x):
    Nch=x[:,0].size
    Nsamp=x[0,:].size

    phase_shifts=2*np.pi*np.arange(0,Nch,1)/Nch
    phase_shifts=np.expand_dims(phase_shifts,axis=1)
    phase_shifts=np.repeat(phase_shifts,Nsamp,axis=1)

    return x*np.exp(-1j*phase_shifts)
