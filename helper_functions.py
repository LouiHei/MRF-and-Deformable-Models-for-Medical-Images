# All functions used in Nerve_segmentation.ipynb
import numpy as np
import skimage
import maxflow
import scipy 

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


def DeformeableModel(im, curve, alpha, beta, stepsize, maxiter):
    l = curve.shape[0]

    # 0. Make the smoothing kernel once
    B = np.zeros((l,l))
    
    # Kernel for minimum length
    Lml = -2*np.identity(l, dtype = None).astype(np.int); 
    Lml += np.roll(np.identity(l, dtype = None).astype(np.int),1,axis = 0)
    Lml += np.roll(np.identity(l, dtype = None).astype(np.int),-1,axis = 0)
    # Kernel for minimum curvature
    Lmc = -6*np.identity(l, dtype = None).astype(np.int); 
    Lmc += np.roll(-np.identity(l, dtype = None).astype(np.int),2,axis = 0)
    Lmc += np.roll(-np.identity(l, dtype = None).astype(np.int),-2,axis = 0)
    Lmc += np.roll(4*np.identity(l, dtype = None).astype(np.int),1,axis = 0)
    Lmc += np.roll(4*np.identity(l, dtype = None).astype(np.int),-1,axis = 0)
    # Apply formulae from the course note
    I = np.identity(l, dtype = None).astype(np.int)
    B = np.linalg.inv(I - alpha*Lml - beta*Lmc)
    
    for iteration in range(maxiter):
        # 1. Find Region Of Interest and its inverse mask (= Snake curve interior and exterior area) 
        ROI = []; notROI = []
        # Make set of all image indices:
        i,j = np.indices(im.shape)
        indices = np.asarray([i.flatten(),j.flatten()]) 
        # Divide set of indices into ROI (= inside polygon) or notROI (= outside polygon)
        truth = skimage.measure.points_in_poly(indices.T, curve)
        x = np.where(truth == True)
        ROI.append([indices[0, x], indices[1, x]])
        x = np.where(truth == False)
        notROI.append([indices[0, x], indices[1, x]])
        ROI = np.asarray(ROI)
        notROI = np.asarray(notROI)
    
        # 2. Compute bilinear interpolaten of the Snake curve:
        I = np.zeros((l,1))
        x0 = np.floor(curve[:,0]).astype(np.int)
        x1 = np.ceil(curve[:,0]).astype(np.int)
        y0 = np.floor(curve[:,1]).astype(np.int)
        y1 = np.ceil(curve[:,1]).astype(np.int)
        dx = curve[:,0] - x0
        dy = curve[:,1] - y0
        I[:,0] = im[y0,x0]*(1 - dx)*(1 - dy) + im[y0,x1]*dx*(1 - dy) + im[y1,x0]*(1 - dx)*dy + im[y1,x1]*dx*dy

        # 3. Using these Regions Of Interest, compute the magnitude (scalar value) of the Snake displacement:
        # Fext =(m_in − m_out)*(2*I − m_in − m_out)*N
        # Find mean intensities as before with ROI:
        m_in  = np.mean(im[ROI])
        m_out = np.mean(im[notROI])
        F = (m_in - m_out)*(2*I - m_in - m_out)

        # 4. Find normal vectors of the Snake curve:
        NormVecs = np.zeros((l,2))
        TangVecs = np.zeros((l,2))
        # Calculate tangent vectors with finite dfference
        TangVecs =  np.roll(curve,-1, axis = 0) - np.roll(curve,1, axis = 0)
        TangVecs[-1,:] = TangVecs[0,:]
        # Normalize lengths
        lengths = (TangVecs[:,0]**2 + TangVecs[:,1]**2)**0.5
        TangVecs[:,0] = TangVecs[:,0]/lengths
        TangVecs[:,1] = TangVecs[:,1]/lengths
        # Compute normal as a 90 degree rotation
        NormVecs[:,0] = TangVecs[:,1]
        NormVecs[:,1] = -TangVecs[:,0]
        # Multiply with correct lengths
        NormVecs = np.multiply(NormVecs, F)

        # 5. Update the curve
        if isinstance(stepsize, int):
            step = stepsize
        else:
            step = stepsize[iteration]
        curve = B @ (curve + step*NormVecs)
        # Redistribute the points evenly
        Dist = (((curve[:,0] - np.roll(curve[:,0],-1))**2 + (curve[:,1] - np.roll(curve[:,1],-1))**2)**0.5)
        correctDist = np.linspace(0,np.sum(Dist),l)
        interpol = scipy.interpolate.interp1d(np.cumsum(Dist), curve[:,0], fill_value="extrapolate")
        curve[:,0] = interpol(correctDist)
        interpol = scipy.interpolate.interp1d(np.cumsum(Dist), curve[:,1], fill_value="extrapolate")
        curve[:,1] = interpol(correctDist)
        # Close the curve
        curve[-1,:] = curve[0,:]
    return curve


############################################
#  Binary segmentation with thresholding 
#   and markov random field graph cuts
############################################

def binary_segmentation(img, threshold=90, beta=0.8, dtype='float'):
    
    # thresholding based on histogram
    img_th = img<threshold

    # Create the graph. If pixel values are int, 
    # pass it in dtype when calling the function
    if dtype == 'float':
        g = maxflow.Graph[float]() 
    else: g = maxflow.Graph[int]()

    # make each pixel of the image a (consecutively labeled) node
    nodes = g.add_grid_nodes(img.shape)

    # Non-terminal edges have the same capacity as the nodes
    g.add_grid_edges(nodes, beta)
    g.add_grid_tedges(nodes, img_th, 1.2-img_th) # terminal edges

    # Compute max. flow through the constructed graph
    g.maxflow()

    # Get the segments of the nodes in the grid.
    # Their labels are 1 where sgm is False and 0 otherwise.
    sgm = g.get_grid_segments(nodes)
    # save the segmented image    
    return np.int_(np.logical_not(sgm))


print('All helper functions loaded.')
