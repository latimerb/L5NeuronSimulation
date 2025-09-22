# L5NeuronSimulation

Analyzing the firing properties of a single, detailed L5b PC neuron.

DOCUMENTATION: https://l5neuronsimulation.readthedocs.io/en/latest/

## Folder Structure

### Core Folders

biophys_components - this is where mechanisms and templates are stored. This folder is used as the basis for all simulations in the project.

FullSimulation - this is the main folder, where the full simulations are run. See README in folder for more.

Group_EPSC_Tuning - where EPSC tuning is performed (on both basal and apical dendrites).

HayFiringProperties - confirming that our cell model matches the experimental firing properties that Hay et al. 2011 tuned for.

IPSC_Tuning - where IPSC tuning is performed (perisomatic, basal, and apical).

MorphAnalysis - folder for exploring the morphology of the cell.

Plot_Cell - folder for plotting the cell and synapse locations on the cell.

ShortPlasticity - used to build and check short term plasticity for both excitation and inhibition.

StepCurrent - used to ensure that a suprathreshold step current induces the proper behavior (as determined by Hay et al. 2011).

