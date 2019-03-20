# All functions used in Nerve_segmentation.ipynb
import numpy as np
import skimage

def icm (segmentation,img,alpha):
    y,x=img.shape
    #mu=np.unique(segmentation)
    mu = [65, 140]
    diffDict=dict()
       
    for k in mu:
        diffNeighborCt=np.zeros((y,x))
        for i,h in [(0,1),(1,0),(-1,0),(0,-1)]:
            tmp = img[1+i:y-1+i,1+h:x-1+h]!=k
            diffNeighborCt[1:y-1,1:x-1] += tmp
            
        diffDict[k]=diffNeighborCt * 4
    vVals=np.zeros((y,x))
    for i in range(y):
        for j in range(x):
            temp=[]
            for k in mu:
                temp.append(alpha*(k-img[i,j])**2+diffDict[k][i,j])
            #temp.append(alpha*(segmentation[i,j]-img[i,j])**2+diffNeighborCt[i,j])
            vVals[i][j]=mu[temp.index(min(temp))]
    return vVals 



# Make the initial Snake a circle:
def InitSnake(center, radius, npoints):
    t = np.linspace(0,2*np.pi, npoints)
    Snake = np.asarray([center[0] + radius*np.cos(t), center[1] + radius*np.sin(t)])
    return Snake

# Find Region Of Interest and its inverse mask (= Snake curve interior and exterior area) 
def FindROI(im, curve):
    # Allocate list names:
    ROI = []; notROI = []
    
    # Make set of all image indices:
    i,j = np.indices(im.shape); indices = np.asarray([i.flatten(),j.flatten()])
    
    # Divide set of indices into ROI (= inside polygon) or notROI (= outside polygon)
    truth = skimage.measure.points_in_poly(indices.T, curve.T)
    for x in range(len(truth)):
        if truth[x] == True:
            ROI.append([indices[0][x], indices[1][x]])
        else:
            notROI.append([indices[0][x], indices[1][x]])
    ROI = np.transpose(ROI); notROI = np.transpose(notROI)
    return ROI, notROI

# Compute bilinear interpolaten of the Snake curve:
def BilinearInterpol(im, curve):
    # Associated intensity after interpolation:
    IntensityOut = np.zeros((1,curve.shape[1]))
    
    for i in range(curve.shape[1]):
        # Declare pixel locations and 'distance' from pixels 
        x  = curve[0,i]
        y  = curve[1,i]
        x0 = np.floor(x).astype(np.int)
        x1 = np.ceil(x).astype(np.int)
        y0 = np.floor(y).astype(np.int)
        y1 = np.ceil(y).astype(np.int)
        dx = x - x0
        dy = y - y0
        IntensityOut[0,i] = im[y0,x0]*(1 - dx)*(1 - dy) + im[y0,x1]*dx*(1 - dy) + im[y1,x0]*(1 - dx)*dy + im[y1,x1]*dx*dy
    return IntensityOut


# Using these Regions Of Interest, Compute the magnitude (scalar value) of the Snake displacement:
# Fext =(m_in − m_out)*(2*I − m_in − m_out)*N
def ComputeDisplacement(im, curve):
    # Find mean intensities as before with ROI:
    ROI, notROI = FindROI(im, curve)
    m_in  = np.mean(im[ROI])
    m_out = np.mean(im[notROI])
    I = BilinearInterpol(im, curve)
    Fext = (m_in - m_out)*(2*I - m_in - m_out)
    return Fext

# Find normal vectors of the Snake curve:
def NormalVectors(im, curve):
    # Allocate space
    NormVecs = np.zeros((2, curve.shape[1]))
    TangVecs = np.zeros((2, curve.shape[1]))
    
    # Calculate tangent vectors with finite dfference
    TangVecs[:,0] = np.asarray([curve[0,1] - curve[0,-1], curve[1,1] - curve[1,-1]])
    TangVecs[:,-1] = TangVecs[:,0]
    for i in range(curve.shape[1]-1):
        TangVecs[:,i] = np.asarray([curve[0,i+1] - curve[0,i-1], curve[1,i+1] - curve[1,i-1]])
        
    # Normalize lengths
    lengths = np.sqrt(TangVecs[0,:]**2 + TangVecs[1,:]**2) 
    TangVecs = np.divide(TangVecs, lengths)
    
    # Compute normal as a 90 degree rotation
    NormVecs[0,:] = TangVecs[1,:]
    NormVecs[1,:] = -TangVecs[0,:]
    
    # Multiply with correct lengths
    F = ComputeDisplacement(im, curve)
    NormVecs = np.multiply(NormVecs, F)
    return NormVecs

def SmoothKernel(curve, alpha, beta):
    l = curve.shape[1]
    B = np.zeros((l,l))
    
    # Construct the transformation matrices:
    # Kernel for minimum length
    Lml = -2*np.identity(l, dtype = None).astype(np.int); 
    for x in range(0,l):
        Lml[x-1,x] = 1
        Lml[x,x-1] = 1

    # Kernel for minimum curvature
    Lmc = -6*np.identity(l, dtype = None).astype(np.int)
    for x in range(0,l):
        Lmc[x-1,x] = 4
        Lmc[x,x-1] = 4
        Lmc[x-2,x] = -1
        Lmc[x,x-2] = -1
        
    I = np.identity(l, dtype = None).astype(np.int)
    B = np.linalg.inv(I - alpha*Lml - beta*Lmc)
    return B


def updateCurve(im, curve, B):
    Vectors = NormalVectors(im, curve)
    # Apply algorithm from the course note: be aware of dimensions n x 2 <--> 2 x n so we hav to (re)transpose!
    curve = B @ (np.transpose(curve + 80*Vectors))
    curve = np.transpose(curve)
    
    # Close the curve
    curve[:,-1] = curve[:,0]
    
    return curve


print('All helper functions loaded.')