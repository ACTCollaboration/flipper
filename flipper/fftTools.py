"""
d@file ffTools.py
@brief FFT  and Power Spectrum Tools
@author Sudeep Das and Tobias A Marriage
"""

import numpy as np
from . import utils
try:
    import pylab
except:
    pass
from numpy.fft import fftshift,fftfreq

import flipper.fft as fftfast

import copy
from scipy.interpolate import splrep, splev
import scipy
import pickle
import sys, os
from .utils import *
from . import flTrace
import astropy.io.fits as pyfits
#import pyfits

__FLIPPER_DIR__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

class fft2D:
    """
    @brief class describing the two dimensional FFT of a liteMap 
    """
    def __init__(self):
        pass

    def copy(self):
        return copy.deepcopy(self)

        
    def mapFromFFT(self,kFilter=None,kFilterFromList=None,showFilter=False,setMeanToZero=False,returnFFT=False):
        """
        @brief Performs inverse fft (map from FFT) with an optional filter.
        @param kFilter Optional; If applied, resulting map = IFFT(fft*kFilter) 
        @return (optinally filtered) 2D real array
        """
        kMap = self.kMap.copy()
        kFilter0 = np.real(kMap.copy())*0.+ 1.
        if kFilter != None:
            
            kFilter0 *= kFilter
            
        if kFilterFromList != None:
            kFilter = kMap.copy()*0.
            l = kFilterFromList[0]
            Fl = kFilterFromList[1] 
            FlSpline = splrep(l,Fl,k=3)
            ll = np.ravel(self.modLMap)
            
            kk = (splev(ll,FlSpline))
            
            kFilter = np.reshape(kk,[self.Ny,self.Nx])
            kFilter0 *= kFilter
        if setMeanToZero:
            id = np.where(self.modLMap == 0.)
            kFilter0[id] = 0.
        #showFilter =  True
        if showFilter:
            pylab.semilogy(l,Fl,'r',ll,kk,'b.')
            #utils.saveAndShow()
            #sys.exit()
            pylab.matshow(fftshift(kFilter0),origin="down",extent=[np.min(self.lx),\
                                                         np.max(self.lx),\
                                                         np.min(self.ly),\
                                                         np.max(self.ly)])
            pylab.show()
        
        kMap[:,:] *= kFilter0[:,:]
        if returnFFT:
            ftMap = self.copy()
            ftMap.kMap = kMap.copy()
            return np.real(fftfast.ifft(kMap,axes=[-2,-1],normalize=True)),ftMap
        else:
            return np.real(fftfast.ifft(kMap,axes=[-2,-1],normalize=True))

    def writeFits(self,file,overWrite=False):
        """
        23-10-2009: added by JB Juin
        02-12-2009: rewrote to include proper WCS keywords (sudeep)
        so that multipoles can be read off in ds9 
        @brief Write a fft2D as a Fits file
        """
        h = pyfits.Header()
        h.update("COMMENT","flipper.fft2D")
        idx = np.where(np.fft.fftshift(self.lx == 0))
        idy = np.where(np.fft.fftshift(self.ly == 0))
        h.update('CTYPE1','ANG-FREQ')
        h.update('CTYPE2','ANG-FREQ')
        h.update("CRPIX1",idx[0][0]+1)
        h.update("CRPIX2",idy[0][0]+1)
        h.update("CRVAL1",0.0)
        h.update("CRVAL2",0.0)
        h.update("CDELT1",np.abs(self.lx[0]-self.lx[1]))
        h.update("CDELT2",np.abs(self.ly[0]-self.ly[1]))
        realFile = file.split('.')[0]+'_real.fits'
        pyfits.writeto(realFile,fftshift(np.real(self.kMap)),header=h,clobber=overWrite)
        del h 
        h = pyfits.Header()
        h.update("COMMENT","flipper.fft2D")
        idx = np.where(np.fft.fftshift(self.lx == 0))
        idy = np.where(np.fft.fftshift(self.ly == 0))
        h.update('CTYPE1','ANG-FREQ')
        h.update('CTYPE2','ANG-FREQ')
        h.update("CRPIX1",idx[0][0]+1)
        h.update("CRPIX2",idy[0][0]+1)
        h.update("CRVAL1",0.0)
        h.update("CRVAL2",0.0)
        h.update("CDELT1",np.abs(self.lx[0]-self.lx[1]))
        h.update("CDELT2",np.abs(self.ly[0]-self.ly[1]))
        realFile = file.split('.')[0]+'_imag.fits'
        pyfits.writeto(realFile,fftshift(np.imag(self.kMap)),header=h,clobber=overWrite)
        
    def trimAtL(self,elTrim):
        """
         @brief Trims a 2-D fft and returns the trimmed fft2D object. Note 
         that the pixel scales are adjusted so that the trimmed dimensions correspond 
         to the same sized map in real-space (i.e. trimming ->
         poorer resolution real space map)
         @pararm elTrim real >0 ; the l to trim at 
         @return fft@D instance
        """
        assert(elTrim>0)
        ft = fft2D()
        idx = np.where((self.lx < elTrim) & (self.lx > -elTrim))
        idy = np.where((self.ly< elTrim) & (self.ly > -elTrim))
        ft.Ny = len(idy[0])
        ft.Nx = len(idx[0])
        
        trimA = self.kMap[idy[0],:]
        trimB = trimA[:,idx[0]]
        ft.kMap = trimB
        del trimA,trimB
        ft.pixScaleX = self.pixScaleX*self.Nx/ft.Nx
        ft.pixScaleY = self.pixScaleY*self.Ny/ft.Ny
        ft.lx = self.lx[idx[0]]
        ft.ly = self.ly[idy[0]]
        ix = np.mod(np.arange(ft.Nx*ft.Ny),ft.Nx)
        iy = np.arange(ft.Nx*ft.Ny)/ft.Nx
        
        modLMap = np.zeros([ft.Ny,ft.Nx])
        modLMap[iy,ix] = np.sqrt(ft.lx[ix]**2 + ft.ly[iy]**2)
        
        ft.modLMap  =  modLMap
    
        
        ft.ix = ix
        ft.iy = iy
        ft.thetaMap = np.zeros([ft.Ny,ft.Nx])
        ft.thetaMap[iy[:],ix[:]] = np.arctan2(ft.ly[iy[:]],ft.lx[ix[:]])
        ft.thetaMap *=180./np.pi
        return ft

    def plot(self,log=False,title='',show=False,zoomUptoL=None):
        """
        @brief Plots an fft2D object as two images, one for the real part and
        another for the imaginary part.
        @param log True means log scale plotting.
        @param title title to put on the plots.
        @param show If True, will show the plots, otherwise create a pylab object
        without showing.
        @param zoomUptoL If set to L, zooms in on the 2-D fft sub-space [-L,L]X[-L,L]
        and then plots.
        @returns Pylab object with plots.
        """
        pReal = fftshift(np.real(self.kMap.copy()))
        pImag = fftshift(np.imag(self.kMap.copy()))
        
        if log:
            pReal = np.log(np.abs(pReal))
            pImag = np.log(np.abs(pImag))
            
        im = pylab.matshow(pReal,origin="down",extent=[np.min(self.lx),np.max(self.lx),\
                                                   np.min(self.ly),np.max(self.ly)])
        pylab.xlabel(r'$\ell_x$',fontsize=15)
        pylab.ylabel(r'$\ell_y$',fontsize=15)
        pylab.colorbar()
        pylab.title(title + '(Real Part)',fontsize=8)
        
        im2 = pylab.matshow(pReal,origin="down",extent=[np.min(self.lx),np.max(self.lx),\
                                                        np.min(self.ly),np.max(self.ly)])
        pylab.xlabel(r'$\ell_x$',fontsize=15)
        pylab.ylabel(r'$\ell_y$',fontsize=15)
        pylab.colorbar()
        pylab.title(title + '(Imaginary Part)',fontsize=8)
                
        if zoomUptoL!=None:
            im.axes.set_xlim(-zoomUptoL,zoomUptoL)
            im.axes.set_ylim(-zoomUptoL,zoomUptoL)
            im2.axes.set_xlim(-zoomUptoL,zoomUptoL)
            im2.axes.set_ylim(-zoomUptoL,zoomUptoL)
        if show:
            pylab.show()

    def getTemplate(self):
        temp = self.copy()
        temp.kMap = np.zeros(self.kMap.shape)
        return temp

def fftTemplateFromPower2D(power2D):
    ft = fft2D()

    ft.Nx, ft.Ny  = power2D.Nx, power2D.Ny
    ft.pixScaleX = power2D.pixScaleX
    ft.pixScaleY = power2D.pixScaleY

    ft.lx, ft.ly  = power2D.lx, power2D.ly
    ft.ix, ft.iy = power2D.ix, power2D.iy

    ft.modLMap  =  power2D.modLMap
    ft.thetaMap =  power2D.thetaMap
    
    ft.kMap     =  np.zeros(power2D.powerMap.shape)
    return ft 
        
def fftFromLiteMap(liteMap,applySlepianTaper = False,nresForSlepian=3.0):
    """
    @brief Creates an fft2D object out of a liteMap
    @param liteMap The map whose fft is being taken
    @param applySlepianTaper If True applies the lowest order taper (to minimize edge-leakage)
    @param nresForSlepian If above is True, specifies the resolution of the taeper to use.
    """
    ft = fft2D()
        
    ft.Nx = liteMap.Nx
    ft.Ny = liteMap.Ny
    flTrace.issue("flipper.fftTools",1, "Taking FFT of map with (Ny,Nx)= (%f,%f)"%(ft.Ny,ft.Nx))
    
    ft.pixScaleX = liteMap.pixScaleX 
                
    
    ft.pixScaleY = liteMap.pixScaleY
    
    
    lx =  2*np.pi  * fftfreq( ft.Nx, d = ft.pixScaleX )
    ly =  2*np.pi  * fftfreq( ft.Ny, d = ft.pixScaleY )
    
    ix = np.mod(np.arange(ft.Nx*ft.Ny),ft.Nx)
    iy = np.arange(ft.Nx*ft.Ny)/ft.Nx
    
    modLMap = np.zeros([ft.Ny,ft.Nx])
    modLMap[iy,ix] = np.sqrt(lx[ix]**2 + ly[iy]**2)
    
    ft.modLMap  =  modLMap
    
    ft.lx = lx
    ft.ly = ly
    ft.ix = ix
    ft.iy = iy
    ft.thetaMap = np.zeros([ft.Ny,ft.Nx])
    ft.thetaMap[iy[:],ix[:]] = np.arctan2(ly[iy[:]],lx[ix[:]])
    ft.thetaMap *=180./np.pi
    
    map = liteMap.data.copy()
    #map = map0.copy()
    #map[:,:] =map0[::-1,:]
    taper = map.copy()*0.0 + 1.0

    if (applySlepianTaper) :
        try:
            path_to_taper = os.path.join(__FLIPPER_DIR__, "tapers", 'taper_Ny%d_Nx%d_Nres%3.1f'%(ft.Ny,ft.Nx,nresForSlepian))
            f = open(path_to_taper)
            taper = pickle.load(f)
            f.close()
        except:
            taper = slepianTaper00(ft.Nx,ft.Ny,nresForSlepian)
            path_to_taper = os.path.join(__FLIPPER_DIR__, "tapers", 'taper_Ny%d_Nx%d_Nres%3.1f'%(ft.Ny,ft.Nx,nresForSlepian))
            f = open(path_to_taper, mode="w")
            pickle.dump(taper,f)
            f.close()
    
    ft.kMap = fftfast.fft(map*taper,axes=[-2,-1])
    del map, modLMap, lx, ly
    return ft


        
class power2D:
    """
    @brief A class describing the 2-D power spectrum of a liteMap
    """
    def __init__(self):
        pass
        
    def copy(self):
        return copy.deepcopy(self)
    
    def powerVsThetaInAnnulus(self,lLower,lUpper,deltaTheta=2.0,powerOfL=0,\
                              fitSpline=False,show=False,cutByMask=False):
        """
        @brief Given an anuulus, radially averages the power spectrum to produce a
        function of angle theta \f$P(\theta)\f$
        @param lLower Lower bound of the annulus
        @param lUpper Upper bound of the annulus
        @param deltaTheta Width of bins in theta
        @param powerOfL The power of L to multiply to PS with, before radial averaging
        @param fitspline If True returns a spline fit to the function
        @param show If True displays the function \f$P(\theta)\f$ 
        @param cutByMask If a kMask exists with p2d, then skip over masked pixels
        
        @return (If fitSpline:) binnedPA, binnedTheta, binCount, binStdDev,logspl,threshold
        
        @return (else:) binnedPA, binnedTheta, binCount, binStdDev 
        """
        if not(cutByMask):
            a= (self.modLMap < lUpper)
            b= (self.modLMap > lLower)
        else:
            a= ((self.modLMap < lUpper)*(self.kMask>0))
            b= ((self.modLMap > lLower)*(self.kMask>0))
        c = a*b
        
        indices = np.where(c)
        p = self.powerMap*self.modLMap**powerOfL
        thetaAnnu = np.ravel(self.thetaMap[indices])
        powerAnnu = (np.ravel(p[indices]))
        
        binnedTheta, binnedPA,binStdDev,binCount = utils.bin(thetaAnnu,powerAnnu,deltaTheta)
        
        if fitSpline:
            logspl = splrep(binnedTheta,np.log(np.abs(binnedPA)),s=2,k=4)
            median = np.median(np.abs(binnedPA))
            
        if show:
            pylab.plot(binnedTheta,np.log(np.abs(binnedPA)),'ro')
            if fitSpline:
                theta = np.arange(180.)
                logSmPA = splev(theta,logspl)
                pylab.plot(theta,logSmPA)
                med = theta.copy()*0. +median
                pylab.plot(theta,np.log(med))
                   
            pylab.xlim(-180.,180.)
            pylab.xlabel(r'$\theta$',fontsize=15)
            pylab.ylabel(r'$\langle \ell^%d C_\ell (\theta)\rangle$'%powerOfL,\
                         fontsize=15)
            pylab.show()
                
        
        
        if fitSpline:
            logspl = logspl
            threshold = median
            return binnedPA, binnedTheta, binCount, binStdDev,logspl,threshold
        else:
            return binnedPA, binnedTheta, binCount, binStdDev
        
    def meanPowerInAnnulus(self,lLower,lUpper,\
                           thetaAvoid1=None,\
                           thetaAvoid2=None,\
                           downweightingSpline=None,\
                           weightMap=None,\
                           threshold=None,\
                           cutByThreshold = False,\
                           cutByMask = False,\
                           showWeight=False,\
                           nearestIntegerBinning=True):
        """
        @brief Given an annulus, takes the mean of the power spectrum in the upper half plane.
        @param lLower Lower bound of the annulus
        @param lUpper Upper bound of the annulus
        @param thetaAvoid1/2 tuples .e.g.[20,60], [120,130] which decides what theta to cut.
        @param downweightingSpline Spline evaluated by power2D.powerVsThetaInAnnulus()
        which is used to weight pixels while adding them.
        @param threshold If present uniform weighting is done when spline is below threshold
        @param cutByThreshold If True, throw out thetas where spline is above threshold.
        @param cutByMask If set to True, multiply the annulus by the k-space mask before adding pixels (see createKspaceMask).
        @return mean,stdDev,number_of_pixels
        """
        lxMap = self.modLMap.copy()*0.
        lyMap = self.modLMap.copy()*0.
        for p in range(self.Ny):
            lxMap[p,:] = self.lx[:]
            
        for q in range(self.Nx):
            lyMap[:,q] = self.ly[:]
        
        lxMap = np.ravel(lxMap)
        lyMap = np.ravel(lyMap)
        if nearestIntegerBinning:
            indices = np.where((np.array(self.modLMap + 0.5, dtype='int64') <= lUpper)\
                                  & (np.array(self.modLMap +0.5, dtype='int64')>=lLower))
        else:
            indices = np.where((self.modLMap <= lUpper)\
                                  & (self.modLMap >lLower))
        
        thetas = np.ravel( self.thetaMap[indices])
        cls =  np.ravel( self.powerMap[indices])
        if cutByMask:
            mask = np.ravel( self.kMask[indices])
        if weightMap!=None:
            linWeightMap = np.ravel(weightMap)
        
        meanP = 0.
        sdP = 0.
        wMeanP = 0.
        wSdP = 0.
        weightsq = 0.
        weight = 0.
        nPix = 0.

        thet=[]
        wet = []
        
        for i in range(len(thetas)):
            
            thisWeight = 1.0
            #if thetas[i] <0.:
            #    continue
            # if lxMap[i] <0 and lyMap[i] == 0: #avoid [-kx,0] because it is same as [kx,0]
            #    continue 
            if thetaAvoid1 != None:
                if thetas[i] >thetaAvoid1[0] and thetas[i] <thetaAvoid1[1]:
                    continue
            if thetaAvoid2 != None:
                if thetas[i] >thetaAvoid2[0] and thetas[i] <thetaAvoid2[1]:
                    continue
            if downweightingSpline != None:
                thisWeight = np.exp(splev(thetas[i],downweightingSpline))
                tw=1./thisWeight**2
                ## because the spline was done on log(y)
                if threshold !=None:
                    if thisWeight < threshold:
                        tw=1./threshold**2
                    if cutByThreshold:
                        if thisWeight > threshold:
                            tw = 0. 
                thisWeight = tw
            if weightMap != None:
                thisWeight = linWeightMap[i]
            if cutByMask:
                if mask[i] == 0:
                    #print "cutting by mask"
                    continue
            
                
            meanP += cls[i]
            wMeanP += cls[i]*thisWeight
            
            sdP += cls[i]**2
            nPix += 1.0
            weight += thisWeight
            weightsq += thisWeight**2
            
            thet.append(thetas[i])
            wet.append(thisWeight)
            
            
        if showWeight:
            pylab.plot(thet,wet,'o')
            pylab.semilogy(thetas,cls/np.max(cls),'o')
            pylab.show()

            
        
        if nPix == 0.:
            #raise ValueError, 'nPix is zero: no observation in bin (%f,%f)'%(lLower,lUpper)
            #print 'nPix is zero: no observation in bin (%f,%f)'%(lLower,lUpper)
            meanP = 1.e-23
            sdP = 1.e-23
        else:
            meanP = meanP/nPix
            sdP = ((sdP - nPix*meanP**2)/(nPix-1))
        if weight != 0:
            wMeanP = wMeanP/weight
            wSdP =np.sqrt( weightsq/weight**2*sdP)
            #if (np.mod(weight,2) == 1.0):
            #    weight += 1.
            weight /= 2
        else:
            wMeanP = 0.
            wSdP = 0.
        
        
        
        #print "weight in bin (%f,%f) is %f"%(lLower,lUpper,weight)
        return wMeanP,wSdP,weight
        #return meanP,sdP

    
        
    def binInAnnuli(self,binningFile,thetaAvoid1=None,thetaAvoid2=None,downweightingSpline=None,\
                    threshold=None,cutByThreshold=None,showWeight=False,\
                    cutByMask = False,\
                    noCutBelowL=None,\
                    weightMap=None,\
                    forceContiguousBins= False,\
                    nearestIntegerBinning = False):
        """
        @brief Bins the 2D power spectrum in L- bins, sepcified by the binningFile
        @param binningFile An ascii file with columns binLower, binUpper, binCenter
        @param noCutBelowL If set to a L -value, overrides all weighting and cuts and
        returns the simple mean in annuli below that L.
        @param other_keywords Same as in meanPowerInAnnulus()
        """
        binLower,binUpper,binCenter = readBinningFile(binningFile)
        nBins = binLower.size
        print("nBins",nBins)
        if forceContiguousBins:
            assert(nearestIntegerBinning == False)
            # redefine bin upper
            binUpper[0:nBins-1] = binLower[1:nBins]
            pass
        
        assert((forceContiguousBins and nearestIntegerBinning) == False)
        
        binMean = np.zeros(nBins)
        binSd =  np.zeros(nBins)
        binWeight = np.zeros(nBins)
        for i in range(nBins):
            if noCutBelowL != None:
                if binUpper[i]<noCutBelowL:
                    (binMean[i],binSd[i]) = self.meanPowerInAnnulus(binLower[i],binUpper[i],\
                                                                    showWeight=showWeight)
                    continue
            (binMean[i],binSd[i],binWeight[i]) = self.meanPowerInAnnulus(binLower[i],binUpper[i],\
                                                                         thetaAvoid1=thetaAvoid1,\
                                                                         thetaAvoid2=thetaAvoid2,\
                                                                         downweightingSpline\
                                                                         =downweightingSpline,\
                                                                         threshold=threshold,
                                                                         cutByThreshold=cutByThreshold,\
                                                                         cutByMask = cutByMask,\
                                                                         showWeight=showWeight,\
                                                                         weightMap=weightMap,\
                                                                         nearestIntegerBinning = nearestIntegerBinning)
        return binLower,binUpper,binCenter,binMean,binSd,binWeight


    def _testIsotropyInAnnulus(self,lLower,lUpper,cutByMask=False):
        """
        @brief Computes the angular integral: \f$ \int \theta P_b(\theta) [\cos^2(2\theta)-1/2] \f$.
        This downweights the 45 degrees directions. The integral should be zero for a truly isotropic spectrum
        @param lLower Lower bound of the annulus
        @param lUpper Upper bound of the annulus
        @param cutByMask If set to True, multiply the annulus by the k-space mask before adding pixels (see createKspaceMask).
        """

        indices = np.where((self.modLMap < lUpper) & (self.modLMap > lLower))
        #print "indices =%s"%str(indices)
        thetas = np.ravel( self.thetaMap[indices])
        cls =  np.ravel( self.powerMap[indices])
        
        if cutByMask:
            mask = np.ravel( self.kMask[indices])
            idx = np.where(mask > 0)
        else:
            idx = np.arange(len(cls))
        
        wtheta = np.cos(2*thetas)**2
        nPix = len(idx)
        
        if nPix>0:
            p = (np.sum(cls[idx]*wtheta[idx]) - (cls[idx].mean())*(wtheta[idx].sum()))/(np.mean(cls[idx]))
        else:
            print('nPix is zero: no observation in bin (%f,%f)'%(lLower,lUpper))
            p = 0.
        return p, nPix
        

    def testIsotropyInAnnuli(self,binningFile,cutByMask=False):
        """
        @brief tests the isotropy in each annulus specfied by the binningFIle (see _testIsotropyInAnnulus)
        @param binningFile An ascii file with columns binLower, binUpper, binCenter (nBins on the first line)
        @param  cutByMask If set to True, multiply the annulus by the k-space mask before adding pixels (see createKspaceMask).
        @return binLowerBound,binUpperBound,BinCenter,BinnedValue,BinWeight
        """
        
        binLower,binUpper,binCenter = readBinningFile(binningFile)
        nBins = binLower.size
        flTrace.issue("fftTools",0, "nBins= %d"%nBins)
        
        
        binMean = np.zeros(nBins)
        binWeight = np.zeros(nBins)
        
        for i in range(nBins):
            (binMean[i],binWeight[i]) = self._testIsotropyInAnnulus(binLower[i],binUpper[i],cutByMask=cutByMask)
            
        return binLower,binUpper,binCenter,binMean,binWeight
        
    def createKspaceMask(self, verticalStripe=None,slantStripeLxLy=None,\
                         slantStripeLxLy2=None,smoothingRadius=None,\
                         apodizeWithGaussFWHM=None):
        """
        @brief Creates a mask in L(K)-space, with Stripes set to zero. Vertical stripes
        are given by [-lx,lx], while slantStripes are specified by the intercepts on the
        X, and Y axes.
        """
        mask = self.powerMap.copy()
        mask[:,:] = 1.
        if verticalStripe!=None:
            idx = np.where((self.lx<verticalStripe[1]) & (self.lx > verticalStripe[0]))
            #print idx
            mask[:,idx] = 0.
        if slantStripeLxLy != None:
            Lx = slantStripeLxLy[0]
            Ly = slantStripeLxLy[1]
            phi = np.arctan(1.0*Ly/Lx)
            #print phi
            perp = Lx*np.sin(phi)
            perpMap = self.modLMap.copy()*np.cos(self.thetaMap*np.pi/180.+phi-np.pi/2.)
            #pylab.imshow(perpMap)
            #pylab.show()
            idxx =np. where(np.abs(perpMap) < np.abs(perp))
            mask[idxx] = 0.
        if slantStripeLxLy2 != None:
            Lx = slantStripeLxLy2[0]
            Ly = slantStripeLxLy2[1]
            phi = np.arctan(1.0*Ly/Lx)
            #print phi
            perp = Lx*np.sin(phi)
            perpMap = self.modLMap.copy()*np.cos(self.thetaMap*np.pi/180.+phi-np.pi/2.)
            #pylab.imshow(perpMap)
            #pylab.show()
            idxxx =np. where(np.abs(perpMap) < np.abs(perp))
            mask[idxxx] = 0.
        if smoothingRadius!=None:
            mask = fftshift(blur_image(fftshift(mask),smoothingRadius))

        if apodizeWithGaussFWHM !=None:
            mask *=np.exp(-self.modLMap**2*apodizeWithGaussFWHM**2/(8*np.log(2)))
        self.kMask = mask
        
    def pixelWindow(self):
        """
        @brief the pixel window function
        """
        pixW = np.zeros([self.Ny,self.Nx])
        pixW[self.iy,self.ix] = np.sinc(self.lx[self.ix]*self.pixScaleX/(2.0*np.pi))*\
                                np.sinc(self.ly[self.iy]*self.pixScaleY/(2.0*np.pi))
                              
        pixW = pixW**2

        return pixW

    def divideByPixelWindow(self):
        """
        @brief Divide the power spectrum by the pixel window function
        """
        
        pixW = self.pixelWindow()
        self.powerMap[:] /= pixW[:] 

    
    def trimAtL(self,elTrim):
        """
         @brief Trims a 2-D powerMap and returns the trimmed power2D object. Note 
         that the pixel scales are adjusted so that the trimmed dimensions correspond 
         to the same sized map in real-space (i.e. trimming ->
         poorer resolution real space map)
         @pararm elTrim real >0 ; the l to trim at 
         @return power2D instance
        """
        assert(elTrim>0)
        p2dTrimmed = power2D()
        idx = np.where((self.lx < elTrim) & (self.lx > -elTrim))
        idy = np.where((self.ly< elTrim) & (self.ly > -elTrim))
        p2dTrimmed.Ny = len(idy[0])
        p2dTrimmed.Nx = len(idx[0])
        
        trimA = self.powerMap[idy[0],:]
        trimB = trimA[:,idx[0]]
        p2dTrimmed.powerMap = trimB
        del trimA,trimB
        p2dTrimmed.pixScaleX = self.pixScaleX*self.Nx/p2dTrimmed.Nx
        p2dTrimmed.pixScaleY = self.pixScaleY*self.Ny/p2dTrimmed.Ny
        p2dTrimmed.modLMap = self.modLMap[idy[0],:]
        p2dTrimmed.modLMap = p2dTrimmed.modLMap[:,idx[0]]
        p2dTrimmed.lx = self.lx[idx[0]]
        p2dTrimmed.ly = self.ly[idy[0]]
        p2dTrimmed.thetaMap = self.thetaMap[idy[0],:]
        p2dTrimmed.thetaMap = p2dTrimmed.thetaMap[:,idx[0]]
        if self.kMask !=None:
            p2dTrimmed.kMask = self.kMask[idy[0],:]
            p2dTrimmed.kMask = p2dTrimmed.kMask[:,idx[0]]
        else:
            p2dTrimmed.kMask = None
            
        return p2dTrimmed
        
    
    
    def plot(self,log=False,colorbar=True,title='',powerOfL=0,pngFile=None,show=True,zoomUptoL=None,\
             showMask=False, yrange = None, showBinsFromFile = None,drawCirclesAtL=None,\
             drawVerticalLinesAtL = None, valueRange=None, colorbarLabel=None):
        """
        @brief Display the power spectrum
        """
        #modLMap = self.modLMap
        #modLMap[np.where(modLMap ==0)] = 1.
        p =  self.powerMap.copy()
        
        p[:] *= (self.modLMap[:]+1.)**powerOfL
        p = fftshift(p)
        
        if showBinsFromFile:
            binLower,binUpper,binCenter= readBinningFile(showBinsFromFile)
            theta = np.arange(0,2.*np.pi+0.05,0.05)
            
            for i in range(len(binLower)):
                x,y = binUpper[i]*np.cos(theta),binUpper[i]*np.sin(theta)
                pylab.plot(x,y,'k')

        if drawCirclesAtL !=None:
            for ell in drawCirclesAtL:
                theta = np.arange(0,2.*np.pi+0.05,0.05)
                x,y = ell*np.cos(theta),ell*np.sin(theta)
                pylab.plot(x,y,'k')
                if len(drawCirclesAtL)<5:
                    pylab.text(ell*np.cos(np.pi/4.),ell*np.sin(np.pi/4.),\
                               '%d'%np.int(ell),rotation=-45,horizontalalignment = 'center',\
                               verticalalignment='bottom',fontsize=8)
                    
        if drawVerticalLinesAtL!=None:
            for ell in drawVerticalLinesAtL:
                pylab.axvline(ell)
                
        if log:
            p = np.log10(np.abs(p))

        if yrange != None:
            p[np.where(p < yrange[0])] = yrange[0]
            p[np.where(p > yrange[1])] = yrange[1]
        vmin = p.min()
        vmax = p.max()
        if valueRange != None:
            vmin = valueRange[0]
            vmax = valueRange[1]
        
        im = pylab.imshow(p,origin="down",extent=[np.min(self.lx),np.max(self.lx),\
        np.min(self.ly),np.max(self.ly)],aspect='equal',vmin=vmin,vmax=vmax, interpolation='nearest')
        pylab.title(title,fontsize=8)
        if colorbar:
            cb=pylab.colorbar()
            if colorbarLabel != None:
                cb.set_label(colorbarLabel)
        pylab.xlabel(r'$\ell_x$',fontsize=15)
        pylab.ylabel(r'$\ell_y$',fontsize=15)
        
        if showMask:
            im2 =  pylab.imshow(fftshift(self.kMask.copy()),\
                                origin="down",\
                                extent=[np.min(self.lx),\
                                        np.max(self.lx),\
                                         np.min(self.ly),\
                                         np.max(self.ly)],\
                                aspect='equal')
            pylab.xlabel(r'$\ell_x$',fontsize=15)
            pylab.ylabel(r'$\ell_y$',fontsize=15)
            if colorbar:
                pylab.colorbar()
                
        
        
        if zoomUptoL!=None:
            im.axes.set_xlim(-zoomUptoL,zoomUptoL)
            im.axes.set_ylim(-zoomUptoL,zoomUptoL)
            if showMask:
                im2.axes.set_xlim(-zoomUptoL,zoomUptoL)
                im2.axes.set_ylim(-zoomUptoL,zoomUptoL)
                
            
        
        
                
        
        if show:
            pylab.show()
        if pngFile!=None:
            pylab.savefig(pngFile)
            
    def writeFits(self,file,overWrite=False):
        """
        23-10-2009: added by JB Juin
        12-02-2009: Complete re-write to add WCS info (Sudeep)
        @brief Write a power2D as a Fits file
        """
        h = pyfits.Header()
        h.update("COMMENT","flipper.power2D")
        idx = np.where(np.fft.fftshift(self.lx == 0))
        idy = np.where(np.fft.fftshift(self.ly == 0))
        h.update('CTYPE1','ANG-FREQ')
        h.update('CTYPE2','ANG-FREQ')
        h.update("CRPIX1",idx[0][0]+1)
        h.update("CRPIX2",idy[0][0]+1)
        h.update("CRVAL1",0.0)
        h.update("CRVAL2",0.0)
        h.update("CDELT1",np.abs(self.lx[0]-self.lx[1]))
        h.update("CDELT2",np.abs(self.ly[0]-self.ly[1]))
        pyfits.writeto(file,fftshift(self.powerMap),header=h,clobber=overWrite)
  
    def getTemplate(self):
        temp = self.copy()
        temp.powerMap = np.zeros(temp.powerMap.shape)
        return temp

def readBinnedPower(file):
    """
    @brief reads in a binned power spectrum from a file
    The file must have columns specficed as : binLeft,binRight,l,cl
    """
    binLeft,binRight,l,cl = pylab.load(file,skiprows= 50,unpack=True,usecols=[0,1,2,3])
    return l,cl

def binTheoryPower(l,cl,binningFile):
    """
    @brief Given a theoretical power spectrum l,cl returns a power
    spectrum binned according to the binningFile
    """
    
    binLower,binUpper,binCenter = readBinningFile(binningFile)

    nBins = len(binCenter)
    lBin = np.zeros(nBins)
    clBin =  np.zeros(nBins)
               
    for i in range(len(binCenter)):
        idx = np.where((l<binUpper[i]) & (l>binLower[i]))
        lBin[i] = (l[idx]).mean()
        clBin[i] = (cl[idx]).mean()
        
    return lBin,clBin
    
def plotBinnedPower(lbin,plbin,\
                    minL = 10,\
                    maxL = 10000, \
                    yrange = [0.1,10000],\
                    title = ' ',\
                    pngFile=None,\
                    show=False,\
                    ylog=True,\
                    theoryFile=None,\
                    theoryFactor = 1.0,\
                    yFactor= 1.0,\
                    returnPlot = False,\
                    beamFWHM = 0.,\
                    tag = '',\
                    errorBars=[],\
                    fitNoise=False,\
                    noiseLabel = True,\
                    color='b',\
                    theoryColor='r',\
                    returnNoiseBias=False):

    """
    @brief Plots a binned Power spectrum @todo: More documentation
    """
        
    if theoryFile != None:
        X = pylab.load(theoryFile)
        lth = X[:,0]
        clth = X[:,1]
        clth = theoryFactor*clth*np.exp(-lth*(lth+1.)*(beamFWHM/60.*np.pi/180.)**2/(8.*np.log(2.)))
        
    else:
        lth=[]
        clth=[]
    print(" In fftTools: %f ******* "%yFactor)
    #beamLFactor = exp(-lth*(lth+1.)*(beamFWHM/60.*np.pi/180.)**2/(8.*np.log(2.))) 
    
    
    l2pl2pi =  yFactor*lbin**2*plbin/(2*np.pi)
    if ylog:
        try:
            p1 = pylab.semilogy(lbin,l2pl2pi,'o',label=tag,color=color)
        except:
            pylab.clf()
            pass
        negIndex = np.where(l2pl2pi < 0.)
        if len(negIndex[0])>0:
            #print l2pl2pi[negIndex]
            pylab.semilogy(lbin[negIndex],-l2pl2pi[negIndex],'^',color=color)#, label=tag + ' (-ve)')
        pylab.semilogy(lth,clth,theoryColor)
        
    else:
        pl = pylab.plot(lbin,l2pl2pi,'o',label=tag,color=color)
        pylab.plot(lth,clth,theoryColor)
    if errorBars != []:
        try:
            pylab.errorbar(lbin,l2pl2pi,yerr=yFactor*lbin**2/(2*np.pi)*errorBars,fmt=None)
        except:
            pass
        if ylog:
            if len(negIndex[0])>0:
                print("negIndex **********", negIndex)
                pylab.errorbar(lbin[negIndex],-l2pl2pi[negIndex],\
                               yerr=yFactor*lbin[negIndex]**2\
                               /(2*np.pi)*errorBars[negIndex],fmt=None)
    if fitNoise:
        a = lbin>7000
        b = lbin<14000
        c = plbin > 0.
        c = a*b*c
        index = np.where(c)
        noiseBias = np.mean(plbin[index])*yFactor
        beamFWHM = beamFWHM/60.*np.pi/180.
        deltaT= np.sqrt(noiseBias)/beamFWHM
        
        if noiseLabel:
            #pylab.text(3000,100,\
            #           r'$ \frac{\ell(\ell+1)}{2\pi} \left(\Delta T  \theta_{FWHM}\right)^2 $'\
            #           ,fontsize=18,color='b',rotation = 65.)
            #pylab.text(3000,1500,'$ \Delta T = %4.1f \mu K$'%deltaT,fontsize=18)
            lab =  r'$ \frac{\ell(\ell+1)}{2\pi} \left(\Delta T  \theta_{FWHM}\right)^2;  $'+\
                  r'$ \Delta T \theta_{FWHM} = %4.1f \mu K - \mathrm{arcmin}$'\
                  %(deltaT*beamFWHM*60.*180./np.pi)
            pylab.plot(lbin,lbin**2/(2.*np.pi)*noiseBias,color=color,label=lab)
        
    pylab.xlim(minL,maxL)
    #print yrange
    if yrange == None:
        if ylog:
            pylab.ylim((np.abs(l2pl2pi)).min(),(np.abs(l2pl2pi)).max())
        else:
            pylab.ylim(l2pl2pi.min(),l2pl2pi.max())

    else:
        pylab.ylim(yrange[0],yrange[1])
    
    pylab.xlabel('$\ell$')
    pylab.ylabel('$ \ell(\ell+1) C_\ell/(2\pi) $')
    pylab.title(title)

    if pngFile != None:
        pylab.savefig(pngFile)
    if show:
        pylab.show()
    if returnPlot:
        pass
    else:
        pylab.clf()
    if fitNoise and returnNoiseBias:
        return deltaT


def powerFromFFT(ft,ft2=None):
    """
    @brief Creates a power2D object from ffts.
    @param ft fft2D object
    @param ft2 fft2d object (optional) if present cross power using ft and ft2 is returned,
    otherwise autopower using ft is returned.
    """
    p2d = power2D()
    p2d.Nx = ft.Nx
    p2d.Ny = ft.Ny
    p2d.pixScaleX = ft.pixScaleX
    p2d.pixScaleY = ft.pixScaleY
    p2d.lx = ft.lx
    p2d.ly = ft.ly
    p2d.ix = ft.ix
    p2d.iy = ft.iy
    p2d.modLMap = ft.modLMap
    p2d.thetaMap = ft.thetaMap
    p2d.kMask = None
    mapFFT = ft.kMap.copy()
    mapFFT2 = ft.kMap.copy()
    if ft2!=None:
        mapFFT2 = ft2.kMap.copy()

    area =ft.Nx*ft.Ny*ft.pixScaleX*ft.pixScaleY
    p2d.powerMap = np.real(np.conjugate(mapFFT)*mapFFT2)*area/(ft.Nx*ft.Ny*1.0)**2
    
    return p2d

def powerFromLiteMap(liteMap,liteMap2=None,applySlepianTaper=False,nresForSlepian=3.0):
    """
    @brief Returns the power spectrum of a liteMap or a cross spectrum of two liteMaps
    """
    ft = fftFromLiteMap(liteMap,applySlepianTaper = \
                    applySlepianTaper,
                    nresForSlepian=nresForSlepian)
    if liteMap2 == None:
        p2d = powerFromFFT(ft)
    else:
        ft2 = fftFromLiteMap(liteMap2,applySlepianTaper = \
                    applySlepianTaper,
                    nresForSlepian=nresForSlepian)
        p2d = powerFromFFT(ft2,ft)
    return p2d

def noisePowerFromLiteMaps(liteMap1,liteMap2,applySlepianTaper=True,nresForSlepian=3.0):
    """@brief returns a noise estmate in the first map, by subtracting the cross-spectrum with
    the second map (on the same patch of sky) from its auto-spectrum
    PSNoise =  PS(m1)- PS(m1Xm2) """
    
    ft1 = fftFromLiteMap(liteMap1,applySlepianTaper=applySlepianTaper,
                         nresForSlepian=nresForSlepian)
    
    ft2 = fftFromLiteMap(liteMap2,applySlepianTaper=applySlepianTaper,
                         nresForSlepian=nresForSlepian)
    p2d1 = powerFromFFT(ft1)
    p2d2 = powerFromFFT(ft2)
    p2dx = powerFromFFT(ft1,ft2)

    p2dNoise  = powerFromFFT(ft1,ft2)

    ## p2dNoise.powerMap[:,:] = (p2d1.powerMap[:,:] +\
    ##                               p2d2.powerMap[:,:] -\
    ##                               2*p2dx.powerMap[:,:])/2.0

    p2dNoise.powerMap[:,:] = (p2d1.powerMap[:,:] \
                              - p2dx.powerMap[:,:])
    
    return p2dNoise

def readBinningFile(binningFile):
    """
    @brief reads a binning file.
    Searches for the file in Flipper params dir or the working dir;
    and fails if not found.

    @return binLower
    @return binUpper
    @return binCenter
    """
    
    if not (os.path.exists(binningFile)):

        binningFile = os.path.join(__FLIPPER_DIR__, 'params', binningFile)  

        if not (os.path.exists(binningFile)):
            raise IOError('Binning file %s not found'%binningFile)
        
    binLower,binUpper,binCenter= np.loadtxt(binningFile,skiprows=1,unpack=True)
    return binLower,binUpper,binCenter

def main():
    """
    Do nothing
    """
    pass
