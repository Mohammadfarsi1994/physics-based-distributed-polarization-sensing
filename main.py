"""
Inverse Scattering Algorithm (ISA)
Author: Mohammad Farsi (mohammad.farsi1994@gmail.com)
Last modified: October 23, 2023

-------------------------------------------

This code is an implementation of Inverse Scattering Algorithm (ISA) algorithm based on the paper:
"Polarization-Dependent Loss: New Definition and Measurement Techniques, 2015" by NO ́E et al.

Please note the following considerations when using this code:

1. Sensitivity to Channel Estimation Noise:
   The algorithm is susceptible to channel estimation noise. To mitigate this issue, a learning-based approach has been proposed with higher tolerance to estimation noise. For more information, refer to:
   "Learning to Extract Distributed Polarization Sensing Data from Noisy Jones Matrices" by M. Farsi, C. Häger, M. Karlsson, E. Agrell.

2. Numerical Instability:
   The algorithm experiences numerical instability when the number of segments (N) increases, especially for N > 10. The original authors of the paper are informed

"""
#========================================================#
# imports and constants 
#========================================================#

import torch as tc
import numpy as np
import platform
import subprocess
from components.channelmodel import ChannelModel
from components.invscattering import InverseScattering

def main(num_segments=5):
    # Initialize the device (GPU if available, otherwise CPU)
    device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_freq_samples = num_segments * 2 # at leas num_segments+1
    tau = 1 # DGD normalized to 1

    # Initialize the channel model and get true channel parameters
    chan_model, true_params = initialize_channel_model(num_segments, device)
    ht, hf = chan_model.sample_freq_response(num_freq_samples, true_params)

    # Initialize the inverse scattering model and add noise to the channel response
    inverse_scattering = InverseScattering(num_segments=num_segments, tau=tau)
    ht = add_noise_to_channel(ht)

    # Perform inverse scattering to estimate channel parameters
    h0_est = inverse_scattering.inverse_scattering(ht)
    estimated_params = inverse_scattering.get_params()

    # Show the results
    inverse_scattering.show_results(true_params, estimated_params)
    open_image('results.png')
def initialize_channel_model(num_segments, device):
    # Initialize the channel model with identity matrix and get true channel parameters
    h0_input = tc.eye(2).to(tc.double)
    chan_model = ChannelModel(num_segments=num_segments, h_init=h0_input, device=device)
    true_params = chan_model.gen_rnd_channel_params()
    true_params['gamma'] = 0.07 + 0.1 * true_params['gamma']
    return chan_model, true_params

def add_noise_to_channel(ht, noise_var=0):
    # Add noise to the channel response
    data_temp = np.sqrt(noise_var) * np.random.multivariate_normal(mean=0 * np.ones(2), cov=np.eye(2), size=2 * len(ht))
    noise = tc.tensor(data_temp.reshape(len(ht), 2, 2))
    return ht + noise


def open_image(image_path):
    # Determine the operating system
    if platform.system() == "Windows":
        # For Windows, you can use the 'start' command to open the image
        subprocess.run(["start", image_path], shell=True)
    elif platform.system() == "Linux":
        # For Linux, you can use 'xdg-open' to open the image
        subprocess.run(["xdg-open", image_path])
    elif platform.system() == "Darwin":
        # For macOS, you can use 'open' to open the image
        subprocess.run(["open", image_path])
    else:
        print("Unsupported operating system")




if __name__ == "__main__":
    main()