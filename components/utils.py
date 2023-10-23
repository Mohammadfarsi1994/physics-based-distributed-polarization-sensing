"""This is class of utils.

This class creates useful general functions.
"""

#__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Mohammad Farsi'

import torch as tc
import numpy as np
import matplotlib.pyplot as plt

class Utils():
    
    def __init__(self):
        self.a = 1
    
  
    def wrapToPi(self,phi):
        """Wraps phases to the range [-pi,pi)
        """
        wraped_phase = (phi + np.pi) % (2 * np.pi) - np.pi
        return wraped_phase
    
    def plot_time_freq_response(self, ht, hf):
        fig = plt.figure(layout='constrained', figsize=(4, 3))
        # plt.subplot(1,2,1)
        #c1 = torch.abs(h_rx[:,0,0]+h_rx[:,1,0])
        #c2 = torch.abs(h_rx[:,1,1]+h_rx[:,0,1])
        plt.stem(ht, linefmt='b-', label=r'$|h(t)_{00}|+|h(t)_{01}|$')
        #plt.stem(c2, linefmt='b-', label='|h11|')
        plt.xlabel('time index')
        plt.yscale('log')
        plt.title('Time response')
        plt.legend()

        # plt.subplot(1,2,2)
        # plt.stem(hf, linefmt='b--', label=r'$|h(f)_{10}|+|h(f)_{11}|$')
        # #plt.stem(t2, linefmt='b--', label='|h11|')
        # plt.legend()
        # plt.title('Frquency response')
        # plt.xlabel('freq index')
        return fig    
    
    def plot_tf_response(self, h_true=[],h_net=[], h_ISA=[],labels=['True','Network', 'ISA']):
        plt.figure(layout='constrained', figsize=(4, 3))
        
        if h_true!=[]:
            plt.stem(h_true, linefmt='b--', label=labels[0])
        if h_net!=[]:
            plt.stem(h_net, linefmt='r--', label=labels[1])
        if h_ISA!=[]:  
            plt.stem(h_ISA, linefmt='g--', label=labels[2])   
            
        plt.legend()
        plt.title('Frquency response')
        plt.xlabel('freq index')     
        plt.ylabel(r'$|h(f)_{00}|+|h(f)_{11}|$')

    def plot_freq_response(self, hf_est, hf):
        plt.figure(layout='constrained', figsize=(6, 4))
        plt.stem(hf, linefmt='b--', label=r'$|h(f)_{00}|+|h(f)_{11}|$')
        plt.stem(hf_est.detach().numpy(), linefmt='r-', label=r'$|h(f)_{00}|+|h(f)_{11}|$')
        #plt.stem(t2, linefmt='b--', label='|h11|')
        plt.legend((r'truth',r'est'))
        plt.title('Frquency response')
        plt.xlabel('freq index')    
    
    def plot_channel_params(self,params=[],est_params=[],invscat_param=[], index_to_watch=0, labels=[], keys_to_watch = []):
        
        if keys_to_watch==[]:
            keys_to_watch = params.keys()
            
        i = 1
        n = len(params)
        if est_params==[]:
            fig = plt.figure(layout='constrained', figsize=(15, 3))
            for key in sorted(keys_to_watch):
                if params[key]!=[]:
                    plt.subplot(1,n,i)
                    my_label = labels
                    plt.stem(params[key], linefmt='r-', label= my_label)
                    plt.xlabel('Segment Index')
                    if key =='a':
                        plt.title(rf'$|\cos(\phi)|$ ')
                        plt.ylim(top=1,bottom=0)
                    elif key=='b':
                        plt.title(rf'$|\cos(\psi)|$ ') 
                        plt.ylim(top=1,bottom=0) 
                    elif key=='rotation':
                        plt.title(rf'${key}$ ')   
                    elif key=='gamma':
                        plt.title(rf'$\{key}$ ')  
                        plt.ylim(top=0.5,bottom=0)    
                    else:
                        plt.title(rf'$\{key}$ ')   
                    plt.legend(loc='upper right')
                    i = i+1

        elif  invscat_param == []:
            fig =plt.figure(layout='constrained', figsize=(15, 3))
            for key in sorted(keys_to_watch):
                if params[key]!=[]:
                    plt.subplot(1,n,i)
                    plt.stem(params[key], markerfmt='bo', linefmt='b--', label=labels[0])
   
                    len_val = len(est_params[key])
                    if isinstance(est_params[key], list):
                        plt.stem(est_params[key][index_to_watch], markerfmt='rx', linefmt='r--', label=labels[1])
                    else:
                        plt.stem(est_params[key], markerfmt='rx', linefmt='r--', label=labels[1])
                        
                        
                    plt.xlabel('Segment Index')
                    if key =='a':
                        plt.title(rf'$|\cos(\phi)|$ ')
                        plt.ylim(top=1,bottom=0)
                    elif key=='b':
                        plt.title(rf'$|\cos(\psi)|$ ')
                        plt.ylim(top=1,bottom=0)   
                    elif key=='rotation':
                        plt.title(rf'${key}$ ')  
                    elif key=='gamma':
                        plt.title(rf'$\{key}$ ')  
                        plt.ylim(top=0.5,bottom=0)     
                    else:
                        plt.title(rf'$\{key}$ ')    
                    plt.legend(loc='upper right')
                    i = i+1
        else :
            fig=plt.figure(layout='constrained', figsize=(15, 3))
            for key in sorted(keys_to_watch):
                if params[key]!=[]:
                    plt.subplot(1,n,i)
                    plt.stem(params[key], markerfmt='bo', linefmt='b--', label=labels[0])
   
                    len_val = len(est_params[key])
                    if isinstance(est_params[key], list):
                        plt.stem(est_params[key][index_to_watch], markerfmt='rx', linefmt='r--', label=labels[1])
                        plt.stem(invscat_param[key][index_to_watch], markerfmt='gs', linefmt='g--', label=labels[2])
                    else:
                        plt.stem(est_params[key], markerfmt='rx', linefmt='r--', label=labels[1])
                        plt.stem(invscat_param[key], markerfmt='gs', linefmt='g--', label=labels[2])
                        
                        
                    plt.xlabel('Segment Index')
                    if key =='a':
                        plt.title(rf'$|\cos(\phi)|$ ')
                        plt.ylim(top=1,bottom=0)
                    elif key=='b':
                        plt.title(rf'$|\cos(\psi)|$ ') 
                        plt.ylim(top=1,bottom=0)   
                    elif key=='rotation':
                        plt.title(rf'${key}$ ')  
                    elif key=='gamma':
                        plt.title(rf'$\{key}$ ')  
                        plt.ylim(top=0.5,bottom=0)     
                    else:
                        plt.title(rf'$\{key}$ ')    
                    plt.legend(loc='upper right')
                    i = i+1
        return  fig
    def plot_est_channel_params(self,params=[],est_params=[], index_to_watch=0, labels=[], keys_to_watch = []):
        if keys_to_watch==[]:
            keys_to_watch = params.keys()
        
        n = len(params)
        i = 1
        fig= plt.figure(layout='constrained', figsize=(15, 3))
        for key in sorted(keys_to_watch):
            if params[key]!=[]:
                plt.subplot(1,n,i)
                plt.stem(params[key][index_to_watch], markerfmt='bo', linefmt='b--', label=labels[0])
                plt.stem(est_params[key][index_to_watch], markerfmt='rx', linefmt='r--', label=labels[1])
                plt.xlabel('Segment Index')
                if key =='a':
                    plt.title(rf'$|\cos(\phi)|$ ')
                    plt.ylim(top=1,bottom=0)
                elif key=='b':
                    plt.title(rf'$|\cos(\psi)|$ ') 
                    plt.ylim(top=1,bottom=0)
                elif key=='rotation':
                    plt.title(rf'${key}$ ')
                elif key=='gamma':
                    plt.title(rf'$\{key}$ ')  
                    plt.ylim(top=0.5,bottom=0)   
                else:
                    plt.title(rf'$\{key}$ ')     
                plt.legend(loc='upper right')
                i = i+1
        return fig      
    def plot_param_change(self, ref, est):
        pass
                       
    def gaussian_rv(self, dim=1, mean=0., var=0):
        data_temp = np.sqrt(var)*np.random.multivariate_normal(mean=mean*np.ones(dim), cov = np.eye(dim),size=self.len_freq)
        return tc.tensor(data_temp)     
    
    def append_dict(self, dict_ref,dict_new):
        for key in dict_new:
            dict_ref[key].append(dict_new[key])
            
        return dict_ref
    def sum_dict(self, dict_ref,dict_new):
        for key in dict_new:
            dict_ref[key] = dict_ref[key] + dict_new[key]
            
        return dict_ref
    
    def update(self,frame,a,a_est,b,b_est,ax):
        
        top = 0.5
        bottom = 0.5
        ax[0].clear()  # Clear the previous plot
        ax[1].clear()  # Clear the previous plot
        x_n = a[frame+1]-a[frame]
        y_n = a_est[frame+1]-a_est[frame]
        stem1= ax[0].stem(x_n, markerfmt='bo', linefmt='b-')
        stem2 = ax[0].stem(y_n, basefmt='k', markerfmt='rx', linefmt='r--')
        ax[0].set_xlabel('segment index')
        #ax[0].set_ylabel(r'$|\cos(\phi_{t})|-|\cos(\phi_{t-1})|$')
        ax[0].set_title(r'$|\cos(\phi_{t})|-|\cos(\phi_{t-1})|$')
        ax[0].set_ylim(bottom=-bottom,top=top)
        ax[0].legend(['true','est'])

        x_n = b[frame+1]-b[frame]
        y_n = b_est[frame+1]-b_est[frame]
        stem3= ax[1].stem(x_n, markerfmt='bo', linefmt='b-')
        stem4 = ax[1].stem(y_n,  basefmt='k', markerfmt='rx', linefmt='r--')
        ax[1].set_xlabel('segment index')
        ax[0].set_title(r'$|\cos(\psi_{t})|-|\cos(\psi_{t-1})|$')
        ax[1].set_ylim(bottom=-bottom,top=top)
        ax[1].legend(['true','est'])
        return stem1.stemlines, stem2.stemlines,stem3.stemlines, stem4.stemlines
    def update3(self,frame,a=[],a_est=[],a_invsact=[],b=[],b_est=[],b_invscat=[],ax=[],legend_str=["true",'net','invscat']):
        
        top = 0.5
        bottom = 0.5
        ax[0].clear()  # Clear the previous plot
        ax[1].clear()  # Clear the previous plot
        x_n = a[frame+1]-a[frame]
        y_n = a_est[frame+1]-a_est[frame]
        z_n = a_invsact[frame+1]-a_invsact[frame]
        stem1= ax[0].stem(x_n, markerfmt='bo', linefmt='b-')
        stem2 = ax[0].stem(y_n, basefmt='k', markerfmt='rx', linefmt='r--')
        stem3 = ax[0].stem(z_n, basefmt='k', markerfmt='gs', linefmt='g--')
        ax[0].set_xlabel('segment index')
        #ax[0].set_ylabel(r'$|a_t|-|a_{t-1}|$')
        ax[0].set_title(r'$|\cos(\phi_{t})|-|\cos(\phi_{t-1})|$')
        ax[0].set_ylim(bottom=-bottom,top=top)
        ax[0].legend(legend_str)

        x_n = b[frame+1]-b[frame]
        y_n = b_est[frame+1]-b_est[frame]
        z_n = b_invscat[frame+1]-b_invscat[frame]
        stem4= ax[1].stem(x_n, markerfmt='bo', linefmt='b-')
        stem5 = ax[1].stem(y_n,  basefmt='k', markerfmt='rx', linefmt='r--')
        stem6 = ax[1].stem(z_n,  basefmt='k', markerfmt='gs', linefmt='g--')
        ax[1].set_xlabel('segment index')
        ax[1].set_title(r'$|\cos(\psi_{t})|-|\cos(\psi_{t-1})|$')
        ax[1].set_ylim(bottom=-bottom,top=top)
        ax[1].legend(legend_str)
        return stem1.stemlines, stem2.stemlines,stem3.stemlines, stem4.stemlines, stem5.stemlines,stem6.stemlines
    
    def tikzplotlib_fix_ncols(self, obj):
        """
        workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
        """
        if hasattr(obj, "_ncols"):
            obj._ncol = obj._ncols
        for child in obj.get_children():
            tikzplotlib_fix_ncols(child)
        
    def get_dict(self):
        my_dict = {'gamma': [], 'phi':[], 'psi':[], 'tau':[], 'rotation':[], 'a':[], 'b':[]}
        return my_dict.copy()