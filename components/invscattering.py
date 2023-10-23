"""
This is the Inverse Scattering Class.

The implementation is founded on the research presented in "Polarization-Dependent Loss: New Definition and Measurement Techniques, 2015" by NO ́E et al.

Within this class, the following functionality is provided:
- Accepting the impulse response elements as input, it computes the following parameters for each fiber segment:
    - tau: Parameters pertaining to Differential Group Delay (DGD).
    - psi: Parameters associated with the orientation of polarization.
    - phi: Parameters related to polarization retardation.
    - gamma: Parameters for Polarization-Dependent Loss (PDL).

For more information and to access the source code, please visit the GitHub repository: https://github.com/Mohammadfarsi1994/inverse-scattering
"""

#__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Mohammad Farsi (mohammad.farsi1994@gmail.com)'

import torch as tc
import matplotlib.pyplot as plt
class InverseScattering():
    
    def __init__(self,num_segments = 10, tau = 1, device = 'cpu'):
        self.num_segments = num_segments
        self.len_freq = num_segments+1
        self.tau = tau
        self.num_pol = 2
        self.device = device
        self.gamma_vec = tc.zeros( self.num_segments, device=self.device).to(tc.double)
        self.phi_vec = tc.zeros( self.num_segments, device=self.device).to(tc.double)
        self.psi_vec = tc.zeros( self.num_segments, device=self.device).to(tc.double)
        


    def inverse_scattering(self,ht):
        """Performs inverse scattering algorithm

        Parameters
        ----------
        ht : tensor
            The time response of the channels

        Returns
        -------
        h0                  : tensor (num_segments, 2, 2)
                             The time response of the input (back scattered to find the input)           
        """
        hi  = ht[0:self.num_segments+1,:,:].to(self.device)
        for i in range( self.num_segments, 0, -1):
           
            
            gamma_i = self.compute_PDL(hi,i) # equation (29)
            PDLi = self.__PDL__(-gamma_i)
            
            Li = tc.matmul(PDLi,hi) # equation (30)
            (phi_i,psi_i) = self.__compute_rotation__(Li,i)
            
            SBAi = self.__rotation__(-phi_i,psi_i) 
            Ki = tc.matmul(SBAi,Li) # equation (34)
            Kif = tc.fft.fftn(Ki, dim=(0,)) # frequecny domain
            
            DGDi = self.__DGD__(-self.tau) 
            h_pre_f  = tc.matmul(DGDi,Kif)
            h_pre = tc.fft.ifftn(h_pre_f, dim=(0,), norm="backward")
            
            
            hi = h_pre
            self.gamma_vec[i-1] = gamma_i
            self.phi_vec[i-1] = phi_i
            self.psi_vec[i-1] = psi_i
        
        h0 = hi
          
        return h0
            
    def get_params(self):
        """_summary_

        Returns
        -------
        estimated_params : dict of tensors (4, num_segments) with estimated parameters with keys
                             'gamma'--> PDL parameters
                             'phi' --> retardation parameters
                             'psi' --> orientation parameters
                             'tau' --> None as it is not estimated   
        """
        invscat_est_params = {'gamma':self.gamma_vec.detach()
                             ,'phi': self.phi_vec.detach()
                             ,'psi': self.psi_vec.detach()
                             ,'tau': tc.ones(self.num_segments)}
        return invscat_est_params    
    
    def  compute_PDL(self,ht,n):
        """  Takes ht (channel response) and returns gamma_i the PDL element.
             according to eq(29)

        Parameters
        ----------
        ht : tensor
             Channel response at time n
        n : int
            Time index

        Returns
        -------
        gamma_i = float
                  The PDL element
        """      
        ratio =  (tc.abs(ht[0,0,0])**2 + tc.abs(ht[0,0,1])**2) / (tc.abs(ht[0,1,0])**2 + tc.abs(ht[0,1,1])**2 ) \
                 * (tc.abs(ht[n,0,0])**2 + tc.abs(ht[n,0,1])**2) / (tc.abs(ht[n,1,0])**2 + tc.abs(ht[n,1,1])**2) 
        gamma_i = tc.log(ratio)/4
        return gamma_i
    
    def __PDL__(self,gamma_i = 1):
        """
        Takes gamma_i and returns 2 by 2 PDL matricx.
        according to PDL(\gamma_i) in eq(24)
        """
        pdl = tc.diag(tc.tensor([tc.exp(gamma_i/2), tc.exp(-gamma_i/2)]))
        pdl = pdl.to(tc.cdouble)
        return pdl
    
    def __rotation__(self,phi=0, psi=0):
        """
        Takes phi_i and psi_i and returns the SBA(phi,psi).
        according to SBA(\phi_i, \psi_i) in eq(23)
        """
        rotation_real = tc.tensor([[tc.cos(phi), -tc.sin(psi)*tc.sin(phi)],\
                                   [tc.sin(psi)*tc.sin(phi), tc.cos(phi)]])
        rotation_imag = tc.tensor([[0, tc.cos(psi)*tc.sin(phi)],\
                                   [tc.cos(psi)*tc.sin(phi), 0]])
        rotation = tc.complex(real=rotation_real, imag=rotation_imag)
        
        return rotation
    
    def __DGD__(self,tau = 1):
        """
        Takes gamma_i and returns num_segments by 2 by 2 PDL matricx.
        according to DGD(\tau) in eq(22)
        """
        n = self.num_segments+1
        dgd = tc.empty(n,self.num_pol,self.num_pol,dtype=tc.cdouble)
        for i in range(n):
            temp_tau = tc.tensor(2*tc.pi*tau*i/n)
            dgd_real = tc.tensor([[1, 0],\
                                 [0,tc.cos(temp_tau)]])
            dgd_imag = tc.tensor([[0, 0],\
                                 [0,-tc.sin(temp_tau)]])
            dgd[i] = tc.complex(real=dgd_real, imag=dgd_imag)
            
        return dgd
    
    def __compute_rotation__(self,L_i = tc.ones([10,2,2]), n = 10):
        """
        Takes L_i and returns phi and psi.
        according to eq(32)
        """
        L0 = L_i[0,:,:]
        Ln = L_i[n,:,:]
        
        L0_abs =tc.abs(L0)
        Ln_abs =tc.abs(Ln)
        
        norm_factor = tc.sum(L0_abs**2) + tc.sum(Ln_abs**2)
        
        s0 = tc.dot(L0_abs[1,:],tc.sum(L0_abs**2, dim=0))
        sn = tc.dot(Ln_abs[0,:],tc.sum(Ln_abs**2, dim=0))
        
        
        c0 = tc.dot(L0_abs[:,0],tc.sum(L0_abs**2, dim=0))
        cn = tc.dot(Ln_abs[:,1],tc.sum(Ln_abs**2, dim=0))
        
        
        sin_phi = (s0 + sn)/norm_factor
        cos_phi = (c0 + cn)/norm_factor
        
        #phi = 2*tc.atan2(sin_phi,cos_phi)
        phi = tc.atan(sin_phi/cos_phi)
        
        t0 = tc.dot(L0[0,:], tc.conj(L0[1,:]))
        tn = tc.dot(Ln[0,:], tc.conj(Ln[1,:]))
        psi = tc.angle(tn - t0) - tc.pi/2
        
        return (phi, psi)
    
    
    def show_results(self,true_params, estimated_params):
        plt.figure(layout='constrained', figsize=(12, 4))
        i = 1
        for key in true_params:
            plt.subplot(1,4,i)
            if key=='phi' or key=='psi':
                plt.stem(tc.abs(tc.cos(true_params[key])), linefmt='r-', label='True')
                plt.stem(tc.abs(tc.cos(estimated_params[key])), markerfmt='go', linefmt='b--', label='Inverse Scattering')
                plt.ylabel(rf'$|\cos(\{key})|$')
            else:
                plt.stem(true_params[key], linefmt='r-', label='True')
                plt.stem(estimated_params[key], markerfmt='go', linefmt='b--', label='Inverse Scattering')
                plt.ylabel(rf'$\{key}$')
            plt.xlabel('Segment Index')
            plt.legend(loc='upper right')
            i +=1
        
        plt.savefig('results.png')
        plt.close()
        
