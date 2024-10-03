# pytestbeam
Python based Monte-Carlo simulation of a test beam setup. An EUDET-type Beam telescope is implemented,
but other devices can simply be added. 
As the physics cases are really simplified this repository serves mostly demonstration purposes.
For a more complete simulation of test beam setups take a look at [Allpix Squared](https://allpix-squared.docs.cern.ch/)

The particle beam consists of electrons with a given rate, beam profile, dispersion, angle and total number of particles.
Each particle loose energy according to a Langau distribution and scatters according to a Gaussian with the width calculated by the Highlander formula.
Where the scattering takes place at an infinity thin sheet at the device surface. 
There happens no scattering in the air between the devices.

The devices have rectangular pixels and 'mostly' uniform material budget and are places in the center of the beam, but can be shifted off center.
The clusters are calculated by sampling a Gaussian distribution around the particle intersection point with a width given by the Einstein diffusion formula.
Each device can eater be trigger or untriggered, this only changes part of the data taking as each beam particle has a chance to be triggerd.

<img src="figures/setup_example_events.png" width="450"/>

# Installation
As a requirement **pytestbeam** requires [pylandau](https://github.com/SiLab-Bonn/pylandau), this package runs for now only on Python 3.10.

The package is installed with
```bash
git clone https://github.com/rpartzsch/pytestbeam
cd pytestbeam
pip install -e .
```
# Usage
Create your test beam setup in ```setup.yml```,
execute afterwards:  
```bash
python pytestbeam.py
```
Materials are characterized in ```materials.yml```. To add for example and absorber to the setup add a device with a single row and column pixel with the given material to ```setup.yml```.
The output data is saved as HDF5.