"""
This channel model is based on the work titled "Polarization-Dependent Loss: New Definition and Measurement Techniques, 2015" authored by NO ́E et al.

This class is designed to take the following inputs:
    - The number of segments.
    - An input matrix for the channel, represented as a 2x2 matrix.

It is capable of generating random channel parameters in the form of a dictionary, including:
    - 'tau': Parameters related to Differential Group Delay (DGD).
    - 'psi': Parameters associated with the orientation of polarization.
    - 'phi': Parameters pertaining to polarization retardation.
    - 'gamma': Parameters for Polarization-Dependent Loss (PDL).

Given a dictionary of channel parameters and the number of frequency samples, this class can generate both time and frequency samples of the channel.
"""

__version__ = '0.1'
__author__ = 'Mohammad Farsi (mohammad.farsi1994@gmail.com)'


import torch as tc
import numpy as np
#import complexPyTorch as ctorch

class ChannelModel():
    def __init__(self, num_segments=10, h_init=tc.eye(2), device='cpu'):
        self.device = device
        self.num_segments = num_segments
        self.num_pol = 2
        self.h_init = h_init
        #self.device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
    
    def sample_freq_response(self,num_freq_samples, params):
        # transfering to device
        gamma_vec = params['gamma'].to(self.device)
        phi_vec = params['phi'].to(self.device)
        psi_vec = params['psi'].to(self.device)
        tau_vec = params['tau'].to(self.device)
        #tau = tc.max(tau_vec)
        # ------------------------------------
        #num_freq_samples = int(self.num_segments * tau* 2)
        freq_vec = tc.arange(start=0, end=num_freq_samples, step = 1)/num_freq_samples    
        freq_vec = freq_vec.to(tc.double)
        freq_vec= freq_vec.to(self.device)
        hf = tc.zeros(num_freq_samples, self.num_pol, self.num_pol, dtype=tc.complex128, device=self.device)
        
        # if tau > 1:
        #     num_ifft = int(num_freq_samples/tau)
        # else:
        num_ifft = num_freq_samples
        
        for ifreq in range(num_freq_samples):
            freq = freq_vec[ifreq]
            hf[ifreq,:,:] =  self.freq_response(freq
                                                     , gamma_vec
                                                     , phi_vec
                                                     , psi_vec
                                                     , tau_vec)    
        ht_temp = tc.fft.ifftn(hf,s=(num_ifft,),dim=(0,),norm="backward")    
        #ht = ht_temp[0:self.num_segments + 1, :, :]
        ht = ht_temp
        return (ht, hf)
    
    
    def freq_response(self, freq, gamma_vec, phi_vec, psi_vec, tau_vec):
        # transfer to device
        freq = freq.to(self.device)
        gamma_vec = gamma_vec.to(self.device)
        phi_vec = phi_vec.to(self.device)
        psi_vec = psi_vec.to(self.device)
        tau_vec = tau_vec.to(self.device)
        h_pre = self.h_init.to(self.device)
        #----------------------------------
        h_pre = h_pre.type(dtype=tc.complex128)
        m = self.all_section(freq, gamma_vec, phi_vec, psi_vec, tau_vec) 
        for n in range(self.num_segments):        
            if n==0:
                h_next = tc.matmul(m[n], h_pre)
            else:
                h_next = tc.matmul(m[n], h_next)
        return h_next    
    
    
    def all_section(self, freq =0, gamma=1, phi=0, psi=0, tau=0):
        """_summary_

        Args:
            freq (double): frequncy to evalue the channel H(freq).
            gamma (vector of double tensors): vector of PDL parameters with length num_segments.
            phi  (vector of double tensors): vector of phi parameters with length num_segments.
            psi  (vector of double tensors): vector of psi parameters with length num_segments.
            tau  (vector of double tensors): vector of tau parameters with length num_segments.

        Returns:
            complex tenosr: Calculates PDL(gamma_i)*SBA(phi_,psi_i)*DGD(tau) for all sections at once which corresponds to eq(22)-(24) in the Noe 2015 paper.
                            dimensions (num_segments, num_polarizations=2, num_polarziations=2)
        """
        cos_phi = tc.cos(phi)
        sin_phi = tc.sin(phi)
        sin_psi = tc.sin(psi)
        cos_psi = tc.cos(psi)
        exp_gamma_plus = tc.exp(gamma/2)
        exp_gamma_minus = tc.exp(-gamma/2)
        
        temp_tau = tc.pi*freq*2*tau
        cos_tau = tc.cos(temp_tau)
        sin_tau = tc.sin(temp_tau)
        
        # Calculates PDL(gamma_i)*SBA(phi_,psi_i)*DGD(tau) at once which corresponds to eq(22)-(24) in the Noe 2015 paper.
        real_part = tc.reshape(tc.t(tc.stack([exp_gamma_plus*cos_phi, exp_gamma_plus*sin_phi * (-cos_tau * sin_psi + sin_tau * cos_psi),\
                                exp_gamma_minus * sin_phi * sin_psi, exp_gamma_minus * cos_phi * cos_tau])), (self.num_segments, 2, 2))
        imag_part = tc.reshape(tc.t(tc.stack([tc.zeros(self.num_segments, device=self.device), exp_gamma_plus*sin_phi * (cos_tau * cos_psi + sin_tau * sin_psi),\
                                exp_gamma_minus * sin_phi * cos_psi, -exp_gamma_minus * cos_phi * sin_tau])), (self.num_segments, 2, 2))
        
        mat = tc.complex(real=real_part, imag=imag_part)
        return mat
        
    
    def gen_rnd_channel_params(self):
        """_summary_

        Returns:
            dict: a dictionary of channel parameters with keys {'gamma', 'phi', 'psi', 'tau'}
                    the parameters are initalized randomly exept tau that must be fixed as ISA requires so.
        """
        gamma_vec = tc.rand(self.num_segments).to(tc.double) # PDL exctintion rates -- unifom distribution between [0,1]
        phi_vec =  self.wrapToPi(2*tc.pi*tc.rand(self.num_segments)-tc.pi).to(tc.double) # retardeation parameters -- unifom distribution between [-pi,pi]
        psi_vec =  self.wrapToPi(2*tc.pi*tc.rand(self.num_segments)-tc.pi).to(tc.double) # orientation parameter -- unifom distribution between [-pi,pi]
        tau_vec =  tc.ones(self.num_segments).to(tc.double) # DGD parameter which is assumed fixed for all segments and normalized to 1
        #tau_vec =  tc.rand(self.num_segments).to(tc.double) # The inverse scattering does not allow for different tau for differen sections
        params = {'gamma': gamma_vec.to(self.device), 'phi': phi_vec.to(self.device), 'psi': psi_vec.to(self.device), 'tau': tau_vec.to(self.device)}
        return params
        
    def wrapToPi(self,phi):
        """Wraps phases to the range [-pi,pi)
        """
        wraped_phase = (phi + np.pi) % (2 * np.pi) - np.pi
        return wraped_phase
        

if __name__ == "__main__":
    num_segments = 10 #  number of fiber sections
    num_pol=2
    
    num_freq_samples = num_segments*2 # number of frequency samples (freq distance equal to num_segments/num_freq_samples)
    h_init = tc.eye(2)
        
    chan_model = ChannelModel(num_segments=num_segments, num_pol=num_pol, h_init=h_init)
    true_params = chan_model.gen_rnd_channel_params()
    true_ht, true_hf = chan_model.sample_freq_response(num_freq_samples, true_params)
    print('test done')
