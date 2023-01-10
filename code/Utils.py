import  pandas as pd
import numpy as np

from Machine_Learning_Classifers import MLClassifiers

np.random.seed(42)
import os, shutil
import sys
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    matthews_corrcoef, average_precision_score, precision_recall_curve, auc
import numpy as np
import itertools
from Config_Reader import *
import torch

# System call
# os.system("")

# Class of different styles
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


def specificity_multi_class(pred, target):
        nb_classes=len(np.unique(target))
        pred = torch.tensor(pred)
        target = torch.tensor(target)
        conf_matrix = torch.zeros(nb_classes, nb_classes)
        for t, p in zip(target, pred):
            conf_matrix[t, p] += 1

        TP = conf_matrix.diag()

        sens = 0
        spec=0
        for c in range(nb_classes):
            idx = torch.ones(nb_classes).byte()
            idx[c] = 0
            # all non-class samples classified as non-class
            TN = conf_matrix[
                idx.nonzero()[:,
                None], idx.nonzero()].sum()  # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
            # all non-class samples classified as class
            FP = conf_matrix[idx, c].sum()
            # all class samples not classified as class
            FN = conf_matrix[c, idx].sum()
            sens += (TP[c].item() / (TP[c].item() + FN.item()))
            spec += (TN.item() / (TN.item() + FP.item()))

            # print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
            #     c, TP[c], TN, FP, FN))

        specificity = spec / nb_classes
        sensitivity = sens / nb_classes
        return specificity,sensitivity
def fundamentalEvalMetrics(y_pred, y_test, ind_predictions_prob):

        acc = accuracy_score(y_pred, y_test)
        pre = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = metrics.f1_score(y_test, y_pred, average="weighted")

        pred_prob = []
        pred_prob.extend([r[1] for r in ind_predictions_prob])
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_prob)

        roc = auc(fpr, tpr)

        prec, recl, _ = precision_recall_curve(y_test, pred_prob)
        prc = average_precision_score(y_test, pred_prob)


        if isinstance(y_test, list):
            y_test = np.array(y_test)
        elif isinstance(y_test, np.ndarray):
            y_test = y_test.tolist()
        else:
            y_test = np.array(list(itertools.chain(*y_test)))
        spec,sensitiviy=specificity_multi_class(y_pred,y_test)

        mcc = matthews_corrcoef(y_test, y_pred)
        return acc, pre, rec, f1, spec, mcc, roc,sensitiviy,prc

class ReadAndSplitData:
    def __init__(self, config):
        """
        This module performs
        1) data analysis
        2) data reading
            1) Kfold
            4) Train & Independent
        """
        self.config = config
        self.sequence_type = self.config.data["Encoders"]["sequence_type"]
        self.dataset_path = self.config.data["dataset_file_path"]
        self.independent_testing_file = self.config.data["independent_testing_file"]
        self.random_state = self.config.data["random_state"]
        self.Clear_allFiles = self.config.data["Clear_allFiles"]
        self.data_split_method = self.config.data["data_splitting"]["data_split_method"]
        self.save_results_path=os.path.join(self.config.data["save_results_path"], "Results")

        if self.Clear_allFiles:

            dir = self.save_results_path
            if  os.path.exists(dir):

                for files in os.listdir(dir):
                    path = os.path.join(dir, files)
                    try:
                        shutil.rmtree(path)
                    except OSError:
                        os.remove(path)
            else:
                os.makedirs(dir)

        else:
            dir = self.save_results_path
            if not os.path.exists(dir):

                os.makedirs(dir)

    def fetchDataset(self):
        # there shall only be one method, (2 columner or 3 columner)

        # standard split means dataset is given with one extra column in which train and test labels are given
        if self.data_split_method == "independent":
            return self.fetch_Data_Split()
        else:
            return self.fetch_Data()

    def fetch_Data(self):
        X = None
        Y = None
        Z=None
        from sklearn.preprocessing import LabelEncoder

        df=pd.read_csv(self.dataset_path, names=["seq","class"], skiprows=1)
        # df = df[df.fold == "fold1"]

        X = df["seq"].values.tolist()
        Y = df["class"].values.tolist()

        Y = LabelEncoder().fit_transform(Y)
        return X,Y,Z
    def fetch_Data_Split(self):
        X=None
        Y=None
        Z=None
        from sklearn.preprocessing import LabelEncoder

        df_train = pd.read_csv(self.dataset_path, names=["seq", "class"], skiprows=1)
        df_train["set"] = "train"
        df_test = pd.read_csv(self.independent_testing_file, names=["seq", "class"], skiprows=1)
        df_test["set"] = "test"
        df = pd.concat([df_train, df_test], axis=0)
        X = df["seq"].values.tolist()
        Y = df["class"].values.tolist()
        Z = df["set"].values.tolist()

        Y = LabelEncoder().fit_transform(Y)


        return X, Y, Z


    def Primary_sequence_Reader(self):
        """
        Read the dataset and create dictionary of datasets based on the settings
        dataset[10,0,0,0]=(list_of_sequences , list_of labels, list_of_split_values)
        Returns
        -------
        updated configuration with the list of dictionary

        """
        sequneces_dataset, sequences_labels, dataset_type = self.fetchDataset()

        dataset = (sequneces_dataset, sequences_labels, dataset_type)
        self.config.data.dict["dataset"] = dataset
        return self.config




class forwardMethodSelection:

    def __init__(self,config):
        self.config =config
        self.split=self.config.data["data_splitting"]["data_split_method"]
        self.classifiersResultsDir=os.path.join(self.config.data["save_results_path"],
                                                self.config.data["savepath-estimatorResults_R"])
        self.encoded_enc_SequencesPath = os.path.join(self.config.data["save_results_path"],
                                                      self.config.data["Encoders"]["Encoded_feature_dir"])
        self.selection_metric = "acc"
        self.allencrespath = os.path.join(self.config.data["save_results_path"],
                                     self.config.data["savepath-estimatorResults"])

        # self.mentionedpath=os.path.join(self.config.data["save_results_path"],
        #                                 self.config.data["Evaluation"]["ForwardMethodSelection"]["best_encoder"]["bestencodersavepath"])
        self.savePath=self.classifiersResultsDir
        if not os.path.exists(self.savePath):
            os.mkdir(self.savePath)
        # if not os.path.exists(self.mentionedpath):
        #     os.mkdir(self.mentionedpath)


    def saveEncoding(self, keyperformancecollection, encodingcolection, labels, dataset_Type=None):
        import codecs
        selection_metric=self.selection_metric

        keys=list(keyperformancecollection.keys())
        scores=[]
        for key, value in keyperformancecollection.items():
            scores.append(value[selection_metric])

        maxval=max(scores)
        maxindex=scores.index(maxval)
        best_encoder=encodingcolection[maxindex]
        #already reaading padded ones no need to apply again
        # best_encoder = np.asarray(best_encoder)
        # best_encoder = pad_sequences(best_encoder, padding="post", dtype="float32")
        df = pd.DataFrame(data=best_encoder)

        best_encodername=keys[maxindex]

        # df.insert(loc=-1, value=labels)
        df['labels']=labels

        if dataset_Type==None or len(dataset_Type)==0:
            pass
        else:
            df["set"]=dataset_Type




        savepath=os.path.join(self.mentionedpath, best_encodername+".csv")
        df.to_csv(savepath, index=False)
        print("encoding has been saved in path="+ str(savepath))
    def read_kfold_version_2(self,path):
        ds_type = []
        fold = None

        topperformningdf = pd.read_csv(path, header=None, skiprows=1)
        labels = topperformningdf[topperformningdf.columns[-1]]  # df.iloc[:,-1:]

        labels = np.array(labels.values.tolist()[:])

        topperformningdf.drop(topperformningdf.columns[-1], axis=1, inplace=True)

        return topperformningdf,labels,ds_type,fold
    def read_independent(self,path):
        fold=None
        topperformningdf = pd.read_csv(path, header=None, skiprows=1)
        labels = topperformningdf[topperformningdf.columns[-2]]

        labels = np.array(labels.values.tolist()[:])


        ds_type = topperformningdf[topperformningdf.columns[-1]].values.tolist()
        topperformningdf.drop(topperformningdf.columns[-2], axis=1, inplace=True)
        topperformningdf.drop(topperformningdf.columns[-1], axis=1, inplace=True)
        return topperformningdf,labels,ds_type,fold
    def bruteForceEncodersPropertyEvaluation(self, clf_func, clfsavename, encoder_path):

        self.encodedSequencesPath=encoder_path
        mlc = MLClassifiers(self.config)
        files = [(files) for r, d, files in os.walk(encoder_path)][0]
        allresdf=pd.DataFrame(columns = ["encoder", self.selection_metric])
        # level 1 results..
        for encFeatures in files:
            pt = os.path.join(self.encodedSequencesPath, encFeatures)
            if self.split == "k-fold":
                topperformningdf, labels, ds_type, fold = self.read_kfold_version_2(pt)
            else:

                topperformningdf, labels, ds_type, fold = self.read_independent(pt)

            scoresoftopeperformingclassifier = mlc.ClassifierProvdingDataFrameAtSettingOptimalLevel(clf_func,
                                                                                                    clfsavename,
                                                                                                    np.array(
                                                                                                        topperformningdf),
                                                                                                    labels, ds_type,
                                                                                                    encFeatures,
                                                                                                    self.savePath,
                                                                                                    previousbestperformance=None,
                                                                                                    fold=fold)

            new_row = {'encoder': encFeatures, self.selection_metric:scoresoftopeperformingclassifier[self.selection_metric]}
            allresdf = allresdf.append(new_row, ignore_index=True)

        if len(allresdf) ==0:
            print( "Unable to evaluate as possible combiinations are " + str(
                len(allresdf)))
            print("done with level 2 evaluation")
            sys.exit(0)
            # this setting loop is redundant
        allresdf['encoder'] = allresdf['encoder'].str.replace('  ', ' ')
        topperformingalgofilenames = allresdf
        # topperformingalgofilenames = allresdf[allresdf['encoder'].str.contains(enccmb, regex=False)]
        topperformingalgofilenames = topperformingalgofilenames.sort_values(self.selection_metric,
                                                                            ascending=False)
        topperformingalgofilenames = topperformingalgofilenames[
            'encoder'].tolist()

        if len(topperformingalgofilenames)<1:
            print()

        pt = os.path.join(self.encodedSequencesPath, topperformingalgofilenames[0])
        if self.split == "k-fold":
            topperformningdf, labels, ds_type, fold = self.read_kfold_version_2(pt)
        else:

            topperformningdf, labels, ds_type, fold = self.read_independent(pt)

        scoresoftopeperformingclassifier = mlc.ClassifierProvdingDataFrameAtSettingOptimalLevel(clf_func,clfsavename,
            np.array(topperformningdf), labels, ds_type, topperformingalgofilenames[0], self.savePath,
            previousbestperformance=None, fold=fold)

        best_encodrname = topperformingalgofilenames[0]
        previous_bestPerformance = scoresoftopeperformingclassifier[self.selection_metric]
        bestdf=topperformningdf
        finalresults= {}
        bestencodrname=best_encodrname
        for outerloop in range(0,len(topperformingalgofilenames)):
            curretntiterationres = {}

            for index in range(0, len(topperformingalgofilenames)):
                pt = os.path.join(self.encodedSequencesPath, topperformingalgofilenames[index])

                if topperformingalgofilenames[index] in bestencodrname:

                    continue
                if self.split == "k-fold":

                    nextdf, labels, ds_type, fold = self.read_kfold_version_2(pt)
                else:

                    nextdf, labels, ds_type, fold = self.read_independent(pt)


                combinedDF = pd.concat([pd.DataFrame(bestdf), pd.DataFrame(nextdf)], axis=1, sort=False)
                best_encodrname = bestencodrname + "$" + topperformingalgofilenames[index]


                scores = mlc.ClassifierProvdingDataFrameAtSettingOptimalLevel(clf_func,clfsavename, np.array(combinedDF), labels,ds_type,
                                                                    best_encodrname, self.savePath,
                                                                    previousbestperformance=previous_bestPerformance,
                                                                              fold=fold
                                                                              )

                if scores[self.selection_metric] > previous_bestPerformance:
                    curretntiterationres[best_encodrname] = scores[self.selection_metric]

            if len(curretntiterationres) != 0:

                # yhn py koi error a ra
                curretntiterationressorted = dict(
                    sorted(curretntiterationres.items(), key=lambda x: x[1], reverse=True))
                allbestnames = list(curretntiterationressorted.keys())[0].split("$")
                dfs = []
                for item in allbestnames:
                    pt = os.path.join(self.encodedSequencesPath,item)
                    if self.split == "k-fold":
                        nextdf, labels, ds_type, fold = self.read_kfold_version_2(pt)
                    else:

                        nextdf, labels, ds_type, fold = self.read_independent(pt)

                    dfs.append(nextdf)

                bestdf = pd.concat(dfs, axis=1, sort=False)
                bestperformance = list(curretntiterationressorted.values())[0]
                bestencodrname = list(curretntiterationressorted.keys())[0]

                finalresults[bestencodrname] = list(curretntiterationressorted.values())[0]
            else:

                best_encodrname = bestencodrname
                break

        if len(finalresults) != 0:
            finalresultssorted = dict(
                sorted(finalresults.items(), key=lambda x: x[1], reverse=True))

            names = list(finalresultssorted.keys())[0].split("$")
            res = list(finalresultssorted.values())[0]
            fdfs = []
            for item in names:
                pt = os.path.join(self.encodedSequencesPath, item)

                if self.split == "k-fold":
                    nextdf, labels, ds_type, fold = self.read_kfold_version_2(pt)
                else:

                    nextdf, labels, ds_type, fold = self.read_independent(pt)
                fdfs.append(nextdf)

            optimaldf = pd.concat(fdfs, axis=1, sort=False)
            best_encodrname = list(finalresultssorted.keys())[0]

        else:
            print("best combination is" + best_encodrname)
            optimaldf = bestdf

        print("best", best_encodrname)
        return optimaldf, best_encodrname










