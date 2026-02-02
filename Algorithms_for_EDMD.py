import sys
sys.path.insert(0, "/home/s1903468/Documents/PHD_PROJECT/d3s-master")

import numpy as np
import matplotlib.pyplot as plt

import d3s.domain as domain
import d3s.observables as observables
import d3s.algorithms as algorithms



def find_eigenfunctions_eigenvalues_1D(dataX,dataY,oper,num_evs,xlima,xlimb,num_boxes,sigma,num_mon,type_of_basis,plot_efunctions):
    
    '''This functions estimates the eigenvalues and plots the eigenfunctions (if plot_efunctions variable is set to True)
    of the operator specified by oper. Returns the eigenvalues.
    dataX = initial points of trajectories shape needs to be (1,p), p:number of particles/trajectories
    dataY = final points of trajectories shape needs to be (1,p)
    oper = takes values 'P' for Perron Frobenius and 'K' for Koopman
    num_evs = number of eigenvalues and eigenfunctions to be calculated
    num_boxes = number of boxes in the domain 
    sigma = parameter for the gaussian basis functions
    num_mon = parameter for the number of degree of monomilas used
    type_of_basis = takes values for the type of basis functions used
                    'ind' for indicator functions, 'gaus' for gaussians, 'mon' for monomials
    plot_efunctions = takes value True for the eigenfunctions to be plotted.  
    '''
    
    bounds = np.array([[xlima, xlimb]])
    boxes = np.array([num_boxes])
    Omega = domain.discretization(bounds, boxes)
    # evs = num_evs # number of eigenvalues/eigenfunctions to be computed
    legend_list=['Eigenfunction {}'.format(i+1) for i in range(num_evs)]
    
    
    plt.figure(1,figsize=(9,6))
    if type_of_basis=='ind':
        psi = observables.indicators(Omega)
    if type_of_basis=='gaus':
        psi = observables.gaussians(Omega, sigma=sigma)
    if type_of_basis=='mon':
        psi = observables.monomials(num_mon)
        
    PsiC = psi(Omega.midpointGrid())
    
    _, d, V = algorithms.edmd(dataX, dataY, psi, operator=oper, evs=num_evs) # d:eigenvalues - can be plotted
    if plot_efunctions==True:
        for i in range(num_evs):
            r = np.real(V[:,i].T @ PsiC)
            x = np.linspace(xlima,xlimb,len(r))
            plt.plot(x,r)
        plt.legend(legend_list)
        if oper=='P':
            plt.title('Eigenfunctions of the Perron-Frobenius operator')
        if oper=='K':
            plt.title('Eigenfunctions of the Koopman operator')
        plt.show()
    
    return d




def uniformOnSphere(m):
    '''
    Generate m uniformly sampled data points on the unit sphere.
    '''
    x = np.random.randn(3, m)
    for i in range(m):
        x[:, i] = x[:, i] / np.linalg.norm(x[:, i])
    return x

class voronoiOnSphere(object):
    '''
    Indicator functions for a Voronoi discretization of the unit sphere.
    '''
    def __init__(self, m):
        self.C = self._equidistantOnSphere(m) # centers of Voronoi cells
        self.n_c = self.C.shape[1] # number of cells

    def __call__(self, x):
        d, m = x.shape # d = dimension of state space, m = number of test points
        n = self.n_c # number of basis functions (i.e., number of Voronoi cells)
        D = np.arccos(self.C.T @ x) # geodesic distances between centers and points.
        ind = D.argmin(axis=0) # indices of closest points
        
        y = np.zeros([n, m])
        for i in range(m):
            y[ind[i], i] = 1
        return y

    def __repr__(self):
        return 'Indicator functions for a Voronoi discretization of the unit sphere.'
    
    def _equidistantOnSphere(self, n):
        '''
        Generates approximately n almost equidistant points on the unit sphere.
        
        See https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf.

        '''
        a = 4.0*np.pi / n
        d = np.sqrt(a)
        M_theta = int(np.round(np.pi / d))
        d_theta = np.pi / M_theta
        d_phi = a / d_theta
        x = []
        y = []
        z = []
        for i in range(M_theta):
            theta = np.pi*(i + 0.5) / M_theta
            M_phi = int(np.round(2 * np.pi * np.sin(theta) / d_phi))
            for j in range(M_phi):
                phi = 2*np.pi * j / M_phi
                x.append(np.sin(theta) * np.cos(phi))
                y.append(np.sin(theta) * np.sin(phi))
                z.append(np.cos(theta))
        return np.array([x, y, z])

if __name__ == "__main__":
    X_ = uniformOnSphere(5000) # uniformly sampled test points
    psi = voronoiOnSphere(100) # basis functions
    PsiX = psi(X_)
    
    # plot results
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    for i in range(psi.n_c):
        ind = np.where(PsiX[i, :] == 1)
        ax.scatter(X_[0, ind], X_[1, ind], X_[2, ind])
        
        
def find_eigenfunctions_eigenvalues_sphere_3D(dataX,dataY,oper,num_evs,num_boxes,plot_efunctions):

    psi = voronoiOnSphere(num_boxes)
    xaxis= uniformOnSphere(num_boxes*10)
    PsiX = psi(xaxis)
    
    _, d, V = algorithms.edmd(dataX, dataY, psi, evs=num_evs, operator=oper)
    if plot_efunctions==True:
        for i in range(num_evs):
            A = V[:,i].T @ PsiX
            fig = plt.figure(i+1)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xaxis[0,:], xaxis[1,:], xaxis[2,:], c=A, alpha=0.1)
            if oper=='K':
                ax.set_title(f'Eigenfunction {i+1} of Koopman')
            if oper=='P':
                ax.set_title(f'Eigenfunction {i+1} of Perron Frobenius')
            plt.show() 
            
    return d