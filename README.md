Simulation engine for mitotic chromosome simulation powered by [OpenMM](https://openmm.org) and [Polychrom](https://github.com/open2c/polychrom).\
Supporting sister chromatids and centromere simulations.

## TL;DR Let's just simulate chromosomes and analyze

### Boettiger Servers 
#### Running Simulations
Run Anaconda Powershell Prompt **as administrator**. (`conda` requires admin access). Then activate `tee` virtual environment using the following command:
```Powershell
conda activate tee
```
If activation is successful, your Powershell Prompt should starts with `(tee)`:
```
(tee) PS: [directory]> 
```
Because this virtual environment contains all the required dependencies, we are ready to run simulation. Running simulation is very simple: you just pass the config file (`*.yaml`) that contains hyperparameters for the simulation to the main simulation driver `boettiger-mitotic/main.py`. `main.py` will validate the validity of hyperparameters, coordinate LE and MD simulations, and output the results automatically. `main.py` accepts path to config file through `--config` flag.

To simulate DT40 SMC3-AID Chromosome 1 where the second diagonal happens near 10 Mb without a sister chromatid nor a centromere, run the following command:

Server 1 
```Powershell
python "L:\07_MitoticChromosome\boettiger-mitotic\main.py" --config  "L:\07_MitoticChromosome\boettiger-mitotic\tutorial\boettiger_servers\server_1\config_bothCond.yaml"
``` 

Server 2
```Powershell 
python "F:\Tee\MitoticChromosome\boettiger-mitotic\main.py" --config  "F:\Tee\MitoticChromosome\boettiger-mitotic\tutorial\boettiger_servers\server_2\config_bothCond.yaml"
``` 
This will simulate one chromosome and output it in folder indicated in `out_dir` parameter in the config file. 

On Server 1, it is: 
```
L:\07_MitoticChromosome\Dataset\20251017_Tutorial
```

while on Server 2, it is:
```
F:\Tee\MitoticChromosome\Dataset\20251017_Tutorial
```  

Anecdotally, this simulation should not take more than 10 minutes if your GPU does not share its resources with other GPU-intensive jobs. 

#### Analyzing Results
The 3D positions of monomers are stored in `all_conformations.npy` which contains a `num_MD_timepoints x num_monomers x 3` numpy array. `num_MD_timepoints` = 31, one for each minute, including the initial structure. `num_monomers` = 100,000, where each bead corresponds to 1 kb (Quick math: this simulated chromosome corresponds to 100 Mb in total.)

The 1D positions of Condensins are slightly more complicated to extract, but one should not give up when things are tough. 1D locations are stored in `SMC_pos_0.npy`. Suffix `0` indicates the first sister chromatid (I know, pythonic zero-based indexing. *Tsk* *Tsk*). If your simulation has 2 sisters, `SMC_pos_0.npy` **AND** `SMC_pos_1.npy` both should exist. This file contains a `num_MD_timepoints x num_condensins x 2` numpy array. 

The information about the type and binding status of Condensin resides in `SMC_props_*.npy`. Suffix convention is similar to `SMC_pos_*.npy`. This information is encoded in a `num_MD_timepoints x num_condensins x 4` numpy array. The binding status (`0`=unbound and `1`=bound) is stored in the first entry of the last dimension. On the contrary, the type of Condensin (`0`=Condensin 1 and `1`=Condensin 2) is saved in the last entry of the last dimension of this array. 


See [jupyter notebook](https://github.com/tee-udon/boettiger-mitotic/blob/main/tutorial/boettiger_servers/20251017_AnalyzingResultsTutorial.ipynb) (`tutorial/boettiger_servers/20251017_AnalyzingResultsTutorial.ipynb`) for more information. 



## TODO
### *In silico* experiments 
- [ ] HOOMD integration for equilibrium stalling testing
- [ ] Biophysical pulling experiments

### Software engineering
- [ ] Tutorial for code running
    - [ ] Boettiger servers
    - [ ] Linux HPC
- [ ] Out-of-the-box results for 3D interaction with the polymer
- [ ] Add module for parameter search powered by Bayesian optimization 
- [ ] Clean up the codebase
    - [ ] Add modules for MATLAB export 
    - [ ] Add option for start index in indepnedent replicates 
    - [ ] Smart OpenMM device selection 

### Housekeeping 
- [ ] Write installation guide 
    - [ ] Ensure dependencies met
- [ ] Wrap a pip and conda package
- [ ] Write a wiki for API database 

## Acknowledgements 
This codebase stands on the shoulders of giants, including the following:

Beckwith, K. S., Brunner, A., Morero, N. R., Jungmann, R., & Ellenberg, J. (2025). Nanoscale DNA tracing reveals the self-organization mechanism of mitotic chromosomes. Cell, 188(10). https://doi.org/10.1016/j.cell.2025.02.028  

https://github.com/open2c/polychrom 