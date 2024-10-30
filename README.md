<p align="center">
<img src="https://rumc-gcorg-p-public.s3.amazonaws.com/b/725/LNQ-banner.x10.jpeg" style="margin: 0 auto;">
</p>

# LNQ Challenge 2023: Learning Mediastinal Lymph Node Segmentation with a Probabilistic Lymph Node Atlas

This repository presents the code used for participation in the LNQ 2023 challenge hosted at MICCAI 2023.

Our approach consists of a preprocessing step, the network training and a postprocessing
of the predicted results. In the pre-processing, automatic segmentation of anatomical structures 
and atlas-to-patient registration is performed to generate strong anatomical priors. 
We have used the LNQ2023 train dataset as well as public datasets provided by Roth et al. (2014) with refined 
annotations and Bouget et al. (2020) as training data. nnU-Net serves as segmentation network architecture.

## Inference Instructions

Download the model weights and the probabilistic lymph node atlas from
[IMI Cloud](https://cloud.imi.uni-luebeck.de/s/24q2qr4ycA7CgSP). Unpack *weights.zip* under *Inference* and run the 
Dockerfile.

## Train Instructions

For training the nnU-Net, we need to create three folders: *nnUNet_raw*, *nnUNet_preprocessed* and *nnUNet_results*
(see: [nnUNet dataset formatting](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md})).

### 1. Preprocessing
#### 1.1 Create nnUNet raw

Our preprocessing code, provided in *Inference*, stores preprocessed images in patient-specific folders. 
The resulting images need to be copied in *nnUNet_raw*. The dataset.json file stored within 
*nnUNet_raw* needs to look like this:
```
{ 
 "channel_names": {
   "0": "CT",
   "1": "noNorm",
   "2": "rgb_to_0_1"
 }, 
 "labels": {
   "background": 0,
   "LN": 1
 }, 
 "numTraining": 512, 
 "file_ending": ".nii.gz"
 }
 ```

#### 1.2 Run nnU-Net preprocessing

To start nnU-Net preprocessing replace ```XX``` with the correct dataset number and run this command: 
```
nnUNetv2_plan_and_preprocess -d XX --verify_dataset_integrity -pl ResEncUNetPlanner -c 3d_fullres
```

### 2. Training
First, set the environment variables accoring to 
[nnUNet set environment variables](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md). 
Second, run the training script providing the required arguments.

### 3. Postprocessing

The code for postprocessing can be also found in *Inference*.

## Reference and Citation
Please refer to our work:

```
Sofija Engelson, Jan Ehrhardt, Timo Kepp, Joshua Niemeijer, Heinz Handels (2024). LNQ Challenge 2023: Learning Mediastinal Lymph Node
Segmentation with a Probabilistic Lymph Node Atlas. Machine Learning for Biomedical Imaging, 2, 817–833. URL: https://melba-journal.org/2024:009. 
DOI: https://doi.org/10.59275/j.melba.2024-f95c.
```

BibTex citation:
```
@article{melba:2024:009:engelson,
    title = "LNQ Challenge 2023: Learning Mediastinal Lymph Node Segmentation with a Probabilistic Lymph Node Atlas",
    author = "Engelson, Sofija and Ehrhardt, Jan and Kepp, Timo and Niemeijer, Joshua and Handels, Heinz",
    journal = "Machine Learning for Biomedical Imaging",
    volume = "2",
    issue = "MICCAI 2023 LNQ challenge special issue",
    year = "2024",
    pages = "817--833",
    issn = "2766-905X",
    doi = "https://doi.org/10.59275/j.melba.2024-f95c",
    url = "https://melba-journal.org/2024:009"
}
```

## License
See the LICENSE.txt file for license rights and limitations (CC BY-NC-ND 4.0).