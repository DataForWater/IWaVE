# IWaVE
Image Wave Velocimetry Estimation

This library performs simultaneous analysis of 2D velocimetry and stream depth 
through 2D Fourier transform methods as outlined and demonstrated in

Dolcetti, G., Hortobágyi, B., Perks, M., Tait, S. J. & Dervilis, N. (2022). 
Using non-contact measurement of water surface dynamics to estimate water discharge. 
Water Resources Research, 58(9), e2022WR032829. 
https://doi.org/10.1029/2022WR032829

The code has been based on the original kOmega code developed by Giulio Dolcetti
(University of Trento, Italy) and released on https://doi.org/10.5281/zenodo.7998891

## Installation

To install IWaVE, setup a python (virtual) environment and follow the instructions 
below:

For a direct installation of the latest release, please activate your environment if 
needed, and type

```commandline
pip install iwave
```

To install IWaVE from the source code as developer (i.e. you wish to provide 
contributions to the code), you must checkout the code base with git using an ssh key
authentication. for instructions how to set this up, please refer to 
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

To check out the code and install it, please follow the code below:

```commandline
git clone git@github.com:DataForWater/IWaVE.git
cd IWaVE
pip install -e .
```
This will install the code base using symbolic links instead of copies. Any code 
changes will then immediately be reflected in the installation.

In case you wish to install the code base as developer, and have all dependencies 
for testing installed as well, you can replace the last line by: 

```commandline
pip install -e .[test]
```

You can now run the tests by running:

```commandline
pytest ./tests
```
