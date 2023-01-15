# CCBM_HAR
Combining CCBM and neural density estimation for human activity recognition.

## Running Experiments
To run all the experiments introduced in the thesis please follow the steps below:
  1. Download the three datasets used for evaluation:
     - [Carrot](https://rosdok.uni-rostock.de/resolve/id/rosdok_document_0000010639)
     - [MotionSense](https://github.com/mmalekzadeh/motion-sense/tree/master/data), the file A_deviceMotion_data.zip suffices
     - [UCI HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
  2. In the directory `CCBM_HAR/pytorch-flows/dataloader` create the folder `data`. Extract each of the zip files downloaded in step 1. into the newly created data
  directory. Then rename the folder A_DeviceMotion_data to MotionSense, D2011-KTA-KHY to Carrots and UCI HAR Dataset to UCI HAR. The names have to match, so the
  dataloader can find the files e.g. under the path `CCBM_HAR/pytorch-flows/dataloader/data/Carrots there should be the seven .arff files for each subject of the
  Carrot dataset.
  3. In the directory `CCBM_HAR/pytorch-flows` create a folder named `models` if it doesn't exist already. Here all trained models will be saved. At evaluation
  time if the model with the configuration you are running an experiment on does not exist in the models folder, that model is initalized and trained. If it does
  exist it is laoded from there which significantly speeds up the evaluation. Training all experiment models from scratch for 100 max epochs took around 4 hours
  on my machine, with CUDA enabled on a NVIDA RTX 2070 Super GPU.
  4. Lastly, change the BASEPATH variable in the files `pytorch-flows/dataloader/loader.py`, respectively `pytorch-flows/dataloader/motionSense.py` to the absolute
  path of the dataloader directory. For example in my case `BASEPATH = "C:/Users/bened/PythonWork/CCBM_HAR/pytorch-flows/dataloader"`
  
You can now run all experiments by executing the script `main.py`found in `CCBM_HAR/pytorch-flows`. In that script you can change several hyperparameters of the
learning process or select new experiments you want to run. The results of all experiments you selected are written directly to the Latex file `resultstable.tex`.
Each trained model is saved in the models folder with the naming convention `modelname_datasetname_W_T_A_N_maxepochs.pt`. Using the trained models means they can
just be loaded for evaluation and don't have to be retrained. Specifying a model which does not occur in the models folder means that it will be trained from scratch.
This will take some time depending on the size of the data.
