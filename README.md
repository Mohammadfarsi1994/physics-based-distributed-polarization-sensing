# Inverse Scattering Algorithm:
The code is based on [PyTorch](https://pytorch.org/). 

The implementation is based on "Polarization-Dependent Loss: New Definition and Measurement Techniques, 2015" by Noe et al." 

- The algorithm is susceptible to channel estimation noise
- We have proposed a learning-based approach that has a higher tolerance to estimation noise see ("M. Farsi, C.Häger, M. Karlsson, E. Agrell, "Learning to Extract Distributed Polarization Sensing Data from Noisy Jones Matrices")
- The algorithm experiences numerical instability when the number of segments increases (N>10)

# Required Packages 
- pyTorch
- numpy
- matplotlib

# Usage Example
``` console
main(num_segments = 5)
```
# Additional Information

If you decide to use the source code for your research, please make sure to cite the original paper and ours:

* R. Noé et al, "[Polarization-Dependent Loss: New Definition and Measurement Techniques](https://ieeexplore.ieee.org/abstract/document/6999936)", in Journal of Lightwave Technology, vol. 33, no. 10, pp. 2127-2138, 2015.
* M. Farsi, C. Häger, M. Karlsson, E. Agrell, "[Polarization-Dependent Loss: New Definition and Measurement Techniques]", in Optical Fiber Communication Conference (submitted), 2024.
