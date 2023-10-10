### stage-arthur

## Overview

# datasets folder
Contains the three datasets used to build the first tests on FLAIR as a mobility predictor (nevermind I had to dump Gowalla, so only 2 left).
Only cabspotting and Geolife are used for the p mpc-H algorithm with FLAIR predictor.

# FLAIR prediction folder
Contains the first tests on FLAIR as a mobility predictor.
The notebook is very messy and was just to get a first look of how it looks like.
The first part shouldn't work correctly since Gowalla is no longer in the project.

# optimal-privacy-main folder
Contains the original notebook of the p mpc-H algorithm (p mpc-H old),  the new version with FLAIR as a predictor
(PMRD_final), as well as a python file functions.py containing all functions needed for the new notebook to work.
In PMRD_final, all the important parameters that can be modified are listed in the parameters dictionnary. The next step
is to create a pdf that sums up everything you would want to know about the trajectory. These pdfs are stored alongside in the
folder.

# reports folder
Contains my internship report
Important note: I recently realized that the values I computed to describe the performances of FLAIR as a mobility predictor are wrong. As such, in my internship report, my Table 5.1: FLAIR’s mean errors as a predictor (on one cabspotting trajectory)
does not contain the right results.
I corrected my code in the PMRD_final notebook (The mean errors that are computed correspond to a horizon of 1)



# Contact me
I'd be happy to answer to any question you may have on the code/ reports
mail: arthur.goarant@centrale.centralelille.fr


