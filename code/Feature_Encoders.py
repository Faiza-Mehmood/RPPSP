import ast
import codecs
import itertools
import shutil
from collections import OrderedDict
from pathlib import Path
import numpy
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import StratifiedKFold
from igraph import *
import re
from Config_Reader import Config
from Utils import *
np.random.seed(42) # cpu vars
import random
random.seed(42)
import pickle
from math import *
from Machine_Learning_Classifers import *
from keras_preprocessing.sequence import pad_sequences



baseSymbol = 'ARNDCQEGHILKMFPSTWYV'

def CalculateMatrix(data, order, base, ktuple):

    n_ncleotides = len(list(itertools.product(base, repeat=ktuple)))


    matrix = np.zeros((len(data[0]) - ktuple+1, n_ncleotides))
    for i in range(len(data[0]) - ktuple+1): # position
        for j in range(len(data)):
            if re.search('-', data[j][i:i+ktuple]):
                pass
            else:
                matrix[i][order[data[j][i:i+ktuple]]] += 1
    return matrix

BASE_DIR=Path(__file__).resolve().parent.parent
class FeatureEncodingApparoaches:
    proteinElements = 'ARNDCQEGHILKMFPSTWYV'
    def __init__(self,config):
        self.config = config
        self.featureEncoders = self.config.data["Encoders"]["feature_encoders"]
        self.sequence_type = self.config.data["Encoders"]["sequence_type"]
        self.kfold = self.config.data["data_splitting"]["k_fold"]["n_folds"]
        self.split = self.config.data["data_splitting"]["data_split_method"]
        self.dataset=self.config.data["Prediction"]["modification_type"]
        self.dt_type=self.config.data["Prediction"]["Dataset_name"]
        self.lambdain = self.config.data["Encoders"]["parameters"]["lambdain"]
        self.weight = self.config.data["Encoders"]["parameters"]["weight"]
        self.L_true = self.config.data["Encoders"]["parameters"]["L_true"]
        if self.L_true:
            self.L_values = self.config.data["Encoders"]["parameters"]["L_values"]
        else:
            self.L_values = [1]
        self.dataset_path = self.config.data["dataset_file_path"]
        self.save_results_path = os.path.join(self.config.data["save_results_path"])
        self.Encoded_feature_dir = os.path.join(self.save_results_path, "Results",
                                                self.config.data["Encoders"]["Encoded_feature_dir"])
        if not os.path.exists(self.Encoded_feature_dir):
            os.mkdir(self.Encoded_feature_dir)


    # function for getting the sequence type
    def sequenceType(self, seqType):

        elements = FeatureEncodingApparoaches.proteinElements

        return elements
    # function for generating and saving the encoded features
    def generate_features(self,encoders=None):
        self.sequences_dataset=self.config.data["dataset"][0]
        self.sequences_labels=self.config.data["dataset"][1]
        self.dataset_type=self.config.data["dataset"][2]


        res = self.selectFeatures("st")
        if res == "error":
            print(res)

  # Function for getting the feature encoder functions
    @staticmethod
    def getFeatureEncoders( sequenceType):
        supportedFeatureEncodersPROT=dict()

        supportedFeatureEncodersPROT['lambdaweight'] = ['CTAPAAC']

        return supportedFeatureEncodersPROT


    def CTAPAAC(self, sequence, lambdaValue=30, w=0.05, L_value=1, props=None):
        if props == "default":
            props = ["Hydrophobicity", "Hydrophilicity", "SideChainMass"]
        if len(sequence) < lambdaValue + 1:
            print(
                'Error: all the sequence length should be larger than the lambdaValue+1: ' + str(
                    lambdaValue + 1) + '\n\n')
            return 0
        with open(os.path.join(BASE_DIR, 'PhysioChemical_Properties/PAAC.txt')) as f:
            AA = "".join(f.readline().split('\t')[1:]).replace("\n", "")
            records = f.readlines()
        for chars in sequence:
            if chars not in AA:
                sequence = sequence.replace(chars, "")

        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i
        AAProperty = []
        AAPropertyNames = []
        for i in range(len(records)):
            array = records[i].rstrip().split() if records[i].rstrip() != '' else None
            if props:
                if array[0] in props:
                    AAProperty.append([float(j) for j in array[1:]])
                    AAPropertyNames.append(array[0])
            else:

                AAProperty.append([float(j) for j in array[1:]])
                AAPropertyNames.append(array[0])

        AAProperty1 = []
        code = []
        for i in AAProperty:
            meanI = sum(i) / 20
            fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
            AAProperty1.append([(j - meanI) / fenmu for j in i])
            theta = []
            for n in range(1, lambdaValue + 1):
                for j in range(len(AAProperty1)):
                    theta.append(
                        sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                             range(len(sequence) - n)]) / (len(sequence) - n))
            myDict = {}
            myDict2 = {}
            myDict3 = {}
            myDict4 = {}
            myDict5 = {}

            H = numpy.zeros((20, len(sequence))).astype(int)
            L = L_value  # size of the patch（补丁） used for extracting the descriptors
            seq = sequence

            Len = len(seq)

            for i in range(Len):
                t = 0
                if seq[i] == 'A':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'R':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'N':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'D':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'C':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'Q':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'E':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'G':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'H':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'I':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'L':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'K':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'M':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'F':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'P':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'S':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'T':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'W':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'Y':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

                t = t + 1
                if seq[i] == 'V':
                    H[t, i] = 1
                else:
                    H[t, i] = 0

            F = numpy.zeros((20, L * 2))  # for 0 and 1 fopr each subsequences

            # 揷omposition, " i.e., the frequency of 0s and 1s
            for i in range(20):  # the 10 binarysequence H

                S = max([len(seq) / L, 1])
                t = 0
                for j in range(1, L + 1):

                    F[i, t] = round(list(H[i, :])[floor((j - 1) * S): floor(j * S)].count(1) / S, 4)
                    t = t + 1
                    # F[i, t] = round(list(H[i,:])[ floor((j - 1)  * S): floor((j) * S)-1].count(0)/S,4)

                    if j == 1:
                        F[i, t] = round(list(H[i, :])[floor((j - 1) * S): floor((j) * S)].count(0) / S, 4)
                    else:
                        F[i, t] = round(list(H[i, :])[floor((j - 1) * S): floor((j) * S) - 1].count(0) / S, 4)
                    t = t + 1

            # ransition? i.e., the percent of frequency with which 1 is followed by 0 or 0 is followed by 1 in a characteristic sequence

            F1 = [0] * 20  # for 0-1 transition, 1 , 11, 111 fopr each subsequences

            for i in range(20):  # the 10 binarysequence H
                S = max([len(seq) / L, 1])
                t = 0

                temp = []
                for j in range(1, L + 1):
                    Sezione = list(H[i, :])[floor((j - 1) * S): floor((j * S) - 1)]
                    Sezione1 = Sezione[1:len(Sezione)]
                    Sezione2 = Sezione[2:len(Sezione)]

                    # print(Sezione)
                    # print(Sezione1)

                    counter1 = 0
                    counter2 = 0
                    counter3 = 0
                    for k in range(len(Sezione1)):
                        if Sezione[k] == 1 and Sezione1[k] == 0:
                            counter1 += 1
                        if Sezione[k] == 0 and Sezione1[k] == 1:
                            counter1 += 1

                    for k in range(len(Sezione1)):

                        if Sezione[k] == 1 and Sezione1[k] == 1:
                            counter2 += 1

                    for k in range(len(Sezione2)):

                        if Sezione[k] == 1 and Sezione1[k] == 1 and Sezione2[k] == 1:
                            counter3 += 1

                    # print(counter1)
                    # print(counter2)
                    # print(counter3)
                    temp.extend([counter1, counter2, counter3])

                F1[i] = temp

            F1 = numpy.array(F1).astype("float")

            nuc_trans = numpy.concatenate((F, F1), axis=1)
            base_char = "ARNDCQEGHILKMFPSTWYV"
            index = 0
            for aa in base_char:
                myDict[aa] = (nuc_trans[index][0])
                myDict2[aa] = (nuc_trans[index][1])
                myDict3[aa] = (nuc_trans[index][2])
                myDict4[aa] = (nuc_trans[index][3])
                myDict5[aa] = (nuc_trans[index][4])
                index += 1


            code = code + ([myDict[aa] / (1 + w * sum(theta)) for aa in AA])
            code = code + ([myDict2[aa] / (1 + w * sum(theta)) for aa in AA])
            code = code + ([myDict3[aa] / (1 + w * sum(theta)) for aa in AA])
            code = code + ([myDict4[aa] / (1 + w * sum(theta)) for aa in AA])
            code = code + ([myDict5[aa] / (1 + w * sum(theta)) for aa in AA])
            code = code + ([w * value / (1 + w * sum(theta)) for value in theta])

        t = []
        for t1 in code:
            t.append(t1)
        return t

    # function for saving the generated encodings into the file
    def saveEncoding(self, featureencoder, labels, additonalParam, encoding,dataset_Type=None):
        import codecs
        savedir=self.Encoded_feature_dir
        filename=featureencoder+"-"+additonalParam+".csv"
        savePath=os.path.join(savedir, filename)


        encoding=np.asarray(encoding)
        encoding = pad_sequences(encoding, padding="post", dtype="float32")
        df=pd.DataFrame(data=encoding)
        # df['seqEncoded']=encoding
        df['labels']=labels

        if dataset_Type!=None:
            df["set"]=dataset_Type

        # df['seqEncoded'] = df['seqEncoded'].str[0].astype(float)
        # df['label'] = df['label'].str[0].astype(float)

        df.to_csv(savePath, index=False)
        #
        # for item in range(0, len(encoding)):
        #     for it in encoding[item]:
        #         writer.write(str(it)+" ")
        #     writer.write(","+ str(labels[item]) + "\n")
        # print(str(featureencoder)+"feature vectors have been saved with labels")


    # function for selecting the feature encoder and generatye the features

    def selectFeatures(self, datasetname):
        globals()["CTAPAAC"]=["Hydrophobicity","Hydrophilicity","SideChainMass"]
        try:
            for algo in self.featureEncoders:
                algo_path=os.path.join(self.Encoded_feature_dir, algo)
                if not os.path.exists(algo_path):
                    os.mkdir(algo_path)
                algo_path=algo+"/"+algo


                print(algo)
                encoding = []

                supportedEncoders = FeatureEncodingApparoaches.getFeatureEncoders(self.sequence_type)
                currentEncoderTypewithRespectToarguments = None
                for key, value in supportedEncoders.items():
                    if algo in value:
                        currentEncoderTypewithRespectToarguments = key
                        break

                for prop in globals()[algo]:


                    encoding = []
                    funnrun = 'FeatureEncodingApparoaches.' + algo
                    for index in range(0, len(self.lambdain)):
                        try:
                            for l_val in self.L_values:
                                lam = self.lambdain[index]
                                weight = self.weight[0]
                                encoding = []
                                for sequence, label in zip(self.sequences_dataset, self.sequences_labels):
                                    funnrun = 'FeatureEncodingApparoaches.' + algo
                                    encoding.append(eval(funnrun)(self, sequence, lam, weight,l_val,prop))
                                if encoding is not None:
                                    self.saveEncoding(algo_path + "-" + datasetname,
                                                  self.sequences_labels, "lamweight-" + str(lam)+"-L-"+str(l_val)+"-"+prop, encoding,
                                                  self.dataset_type)
                        except Exception as e:
                            print(style.RED + str(e) + "\n" + "Unable to Encode")
                            print(style.BLACK)
                            continue

                mlc = MLClassifiers(self.config)
                classifiers = dict()
                classifiers["RandomForestClassifier"] = mlc.getCurrentClassifier("RandomForestClassifier")
                classifiers = mlc.update_params(classifiers)
                fs = forwardMethodSelection(self.config)
                optimaldf, best_encoder_name = fs.bruteForceEncodersPropertyEvaluation(
                    classifiers["RandomForestClassifier"], "RandomForestClassifier",
                    os.path.join(self.Encoded_feature_dir, algo))

                names = best_encoder_name.split("$")
                best_encoder = []
                for name in names:
                    # name = name.replace("-res.csv", "")
                    if "hydrophobicity" in name:
                        name=name.replace("hydrophobicity","hy")
                    name = name.replace(".csv", "")
                    if name.split("-")[0] not in best_encoder:
                        best_encoder.append(name.split("-")[0])
                    if name.split("-")[-1]=='':
                        best_encoder.append(name.split("-")[-2])
                    else:
                        best_encoder.append(name.split("-")[-1])
                # best_encoder.append(".csv")
                best_encoder_name="-".join([a for a in best_encoder])

                file_names=os.listdir(os.path.join(self.Encoded_feature_dir,algo))
                # for file_name in file_names:
                #     shutil.move(os.path.join(self.Encoded_feature_dir,algo, file_name), self.Encoded_feature_dir)
                shutil.rmtree(os.path.join(self.Encoded_feature_dir,algo))

                self.saveEncoding(best_encoder_name,
                                  self.sequences_labels,
                                  "",
                                  optimaldf, self.dataset_type)


        except Exception as e:
            print(style.RED + str(e) + "\n" + "Unable to Encode")
            print(style.BLACK)
            return "error"
        return self.Encoded_feature_dir


