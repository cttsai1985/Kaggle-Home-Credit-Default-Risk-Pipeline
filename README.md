## Kaggle-Home-Credit-Default-Risk
This repo aims to kaggling Home Credit Default Risk in a pipeline fashion. This pipeline is not necessary a performer (PB 0.792/ LB 0.796) but, aiming to bring a bit automation and using config files to control its parameters.
It does, 1) data set construction ann caching, 2) feature transform, 3) hyperpparameters search and 4) stacking models to generate submission file.

The necessary data set and descriptions can be fould here. https://www.kaggle.com/c/home-credit-default-risk

### Modules
`ModelPipeline.py` is exposed to excute all the tasks and the other main modules are located in `lib`. In `lib`,
`DataProvider.py` constructs data set and use `FeatureTransformer.py`. `ScikitOptimize.py` use Bayesian Optimization in Scikit-Optimize to perfrom hyperparameter search. `AutoStacker.py` uses mlxtend to perform stacking. The shared varibles and templetes keep in `LibConfigs.py`. Some small tools is implemented in `Utility.py. `DataFileIO.py` is responsible for save and load HDF5 and CSV files.

### Folders
The cached data are stored in `data`, found hyperparameters are stored in `params`, result for submission in `output`.

### Configs
In `configs`, there are three configs to control the whole modeling process:

configs for feature generations -- SampleDataConfigs.py

configs for hyper parameter search -- SampleModelConfigs.py

configs for feature stacking model -- SampleStackerConfigs.py

## Usage
### Load data set and create caches for train and test dataframes
python3 ModelPipeline.py -a cache_prefix

### Re-construct train and test data set, use after modified `*DataConfigs`
python3 ModelPipeline.py --refresh-cache

### Do hyperparamters tuning on base model
python3 ModelPipeline.py --compute-hpo -t LGBM,LossGuideXGB

### Before doing stacking, adding some external results from forked kernels (optional)
python3 LGBMSelectedFeatures.py

### Prepare meta features for stacking
python3 ModelPipeline.py --compute-stack --refresh-meta

### Introduce OOF results from external models (optional)
adding filenames such as `probs_*.hdf5` into variable `ExternalMetaConfigs` in the using `*StackerConfigs.py`

### Last step, do stacking
python3 ModelPipeline.py --compute-stack

### Other flags
--debug
--enable-gpu

