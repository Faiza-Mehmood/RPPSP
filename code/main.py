import os, sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils import *
import os
print(os.getcwd())
import numpy as np
from Feature_Encoders import FeatureEncodingApparoaches
from Machine_Learning_Classifers import MLClassifiers
from Config_Reader import *
np.random.seed(42)
import warnings
warnings.simplefilter("ignore")
BASE_DIR=Path(__file__).resolve().parent.parent
save_path=Path(__file__).resolve().parent.parent.parent
config_path = os.path.join(BASE_DIR, "Config/DataReaderConfig.json")



config = Config(config_path=config_path)





rs = ReadAndSplitData(config)
config = rs.Primary_sequence_Reader()
if config.data["Encoders"]["apply_feature_encoding"]:
    fg = FeatureEncodingApparoaches(config)
    fg.generate_features()
if config.data["Evaluation"]["apply_evaluation"]:
    clf = MLClassifiers(config)
    clf.Level_1_Evaluation()