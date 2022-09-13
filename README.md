# RPPSP:  Robust and Precise Protein Solubility Predictor by utilizing Novel Protein Sequence Encoder (CTAPAAC)


## Requirement

Install requirement by running following command:


    pip3 install -r requirements.txt


## Available Classifiers

* ***list of Classifiers  :*** [ "AdaBoostClassifier", "DecisionTreeClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier", "LogisticRegression", "XGBOOST", "RandomForestClassifier"]

## Configurations

### Experimentation settings

* Data Format: Dataset files must have csv format.
* Framework facilitates two different types of exprimentation: 
    * ***K-fold***
    * ***Independent*** (in this case data is already splitted into train and test set)
* For reading and processing the data, update parameters and paths in ***Config/DataReaderConfig.json***
  * First of all, set dataset path 'dataset_file_path'
  * set 'mode' such as 'training' or 'prediction'
  * set 'Clear_allFiles' such as true or false (if you want to clear the directory)
  * set 'save_results_path' (folder path which contains encoders or all type of results)
  * select Experimentation Critera (k-fold,independent)
  * In case of k-fold experimentation set number of folds 'n_fold' 
### Feature Encoding method Configuration
* set 'parameter' ('lambdain') according to selected lag value
* set 'Encoded_feature_dir' (where you want yo save the encoded data files)
* set 'parameter' "L_true=True" to split a long sequence into defined ('L_values') subsequences. In case of "False" it takes full sequence for computations. 


### Classifiers hyper-parameters Configuration 
* Parameters of classifiers can be change in following file **Config/classifier_config.json**  
### Evaluation Configuration
* Set 'apply_evaluation' true for applying the evaluation on encoder method
* Pass list of classifier in parameter 'classifiers'  for applying classifiers

## Usage

After setting all parameters run main.py file:

    python code/main.py

## Datasets

All Datasets are available [here.](https://drive.google.com/drive/folders/1ZyL9uOhgYYo7l6GPPR8VZOT3U_R_njQQ?usp=sharing)
