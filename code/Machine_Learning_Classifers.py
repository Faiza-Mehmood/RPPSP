import itertools
import pickle
import sys
from collections import Counter
from pathlib import Path
import scipy
import sklearn
import os
import codecs
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import Utils
from Utils import *
import warnings
from Config_Reader import *

# from Utils import EvalutationMetrics
np.random.seed(42)
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 32})
import ast
BASE_DIR2=(Path(__file__).resolve().parent.parent)
class MLClassifiers:

    def __init__(self, config):
        self.config = config
        # self.ev = EvalutationMetrics()
        self.encoded_enc_SequencesPath = os.path.join(self.config.data["save_results_path"],"Results",
                                                      self.config.data["Encoders"]["Encoded_feature_dir"])
        self.predicted_probs = os.path.join(self.config.data["save_results_path"],"Results","Predcited_probs")


        self.split = self.config.data["data_splitting"]["data_split_method"]
        self.kfold = self.config.data["data_splitting"]["k_fold"]["n_folds"]
        self.savePath = os.path.join(self.config.data["save_results_path"],"Results",
                                     self.config.data["savepath-estimatorResults"])
        self.random_state = self.config.data["random_state"]
        self.custom_encoders_names = self.config.data["Encoders"]["feature_encoders"]
        self.multiclass_config_file = self.config.data["Evaluation"]["multiclass_config_file"]
        self.model_path=self.config.data["Prediction"]["model_path"]
        self.dataset = self.config.data["Prediction"]["modification_type"]
        self.dt_type = self.config.data["Prediction"]["Dataset_name"]

        if not os.path.exists(self.savePath):
            os.mkdir(self.savePath)

        if not os.path.exists(os.path.join(self.config.data["save_results_path"],"Results","Actual_Predicted_of_Algos")):
            os.mkdir(os.path.join(self.config.data["save_results_path"],"Results","Actual_Predicted_of_Algos"))
        if not os.path.exists(os.path.join(self.config.data["save_results_path"],"Results","Predcited_probs")):
            os.mkdir(os.path.join(self.config.data["save_results_path"],"Results","Predcited_probs"))

        self.cfpath=os.path.join(self.config.data["save_results_path"],"Results","confusion_matrix")
        if not os.path.exists(os.path.join(self.config.data["save_results_path"],"Results","confusion_matrix")):
            os.mkdir(os.path.join(self.config.data["save_results_path"],"Results","confusion_matrix"))

        self.averageScores = {"acc": 0, "pre": 0, "rec": 0, "f1": 0, "spec": 0,
                              "sensitiviy": 0, "AUPRC": 0, "AUROC": 0, "mcc": 0}
        self.cfmatrix = []
        self.cm_class=[]


    def re_init_paramas(self):
        self.averageScores = {"acc": 0, "pre": 0, "rec": 0, "f1": 0, "spec": 0,
                              "sensitiviy": 0, "AUPRC": 0, "AUROC": 0, "mcc": 0}
        self.actual_test_reg_labels = []
        self.predicted_test_reg_labels = []
        self.predicted_test_reg_prob = []
        self.cfmatrix = []
        self.cm_class=[]

    def save_kfold_results(self):
        self.saveClassfierResults(self.averageScores, str(self.kfold) + "-" + self.classifierSaveName + "-" + (os.path.basename(self.encFeaturesPath).strip()))

        np.save(self.predicted_probs + "/" + str(self.kfold) +"-"+ "pred-" +  "-" + self.classifierSaveName + "-" + os.path.basename(self.encFeatures), self.predicted_test_reg_prob)
        np.save(self.predicted_probs + "/" +str(self.kfold) +"-"+ "act-" +  "-"+ self.classifierSaveName + "-"  + os.path.basename(
            self.encFeatures), self.actual_test_reg_labels)

        self.saveActualAndPredicted(self.actual_test_reg_labels, self.predicted_test_reg_labels,
                                    self.predicted_test_reg_prob,
                                    str(self.kfold) + "-" + self.classifierSaveName + os.path.basename(
                                        self.encFeatures),
                                    os.path.join(self.config.data["save_results_path"],"Results","Actual_Predicted_of_Algos"))
        self.namesofAllEncoders.append(
            str(self.kfold) + "-" + self.classifierSaveName + "-" + (os.path.basename(self.encFeaturesPath).strip()))
        self.resultsofAllEncoders.append(self.averageScores)

    def save_ind_results(self):

        np.save(self.predicted_probs + "/" +str(self.split) +"-"+"pred-" +  "-" + self.classifierSaveName + os.path.basename(
            self.encFeatures), self.predicted_test_reg_prob)
        np.save(self.predicted_probs + "/" + str(self.split) +"-"+"act-" +  "-" + self.classifierSaveName + os.path.basename(
            self.encFeatures), self.actual_test_reg_labels)
        tmp_path = os.path.join(self.cfpath,
                                str(self.split) + "-" + self.classifierSaveName + "-" + "" + os.path.basename(
                                    self.encFeatures).strip() + ".png")
        self.saveClassfierResults(self.averageScores,
                                  str(self.split) + "-" + str(self.classifierSaveName) + "-" + self.encFeatures)

        self.saveActualAndPredicted(self.actual_test_reg_labels, self.predicted_test_reg_labels,
                                    self.predicted_test_reg_prob,
                                    str(self.split) + "-" + self.classifierSaveName + os.path.basename(
                                        self.encFeatures),
                                    os.path.join(self.config.data["save_results_path"],"Results", "Actual_Predicted_of_Algos"))
        self.namesofAllEncoders.append(str(self.split) + "-" + self.classifierSaveName + "-" + (os.path.basename(self.encFeaturesPath).strip()))
        self.resultsofAllEncoders.append(self.averageScores)
        print("standard split flow")

    def classify(self):
        try:
            self.clf.fit(self.X_train, self.y_train)
        except:
            try:
                self.X_train = self.X_train.toarray()
                self.X_test = self.X_test.toarray()
                self.y_train = self.y_train.toarray()
                self.y_test = self.y_test.toarray()
                # self.clf.fit(self.X_train, self.y_train)
            except:
                try:
                    self.X_train = self.X_train.todense()
                    self.X_test = self.X_test.todense()
                    self.y_train = self.y_train.todense()
                    self.y_test = self.y_test.todense()
                except:
                    try:
                        self.X_train = np.array(self.X_train)
                        self.X_test = np.array(self.X_test)
                        self.y_train = np.array(self.y_train)
                        self.y_test = np.array(self.y_test)
                        self.clf.fit(self.X_train, self.y_train)
                    except:
                        try:
                            self.X_train = sparse.csr_matrix(self.X_train)
                            self.X_test = sparse.csr_matrix(self.X_test)
                            # self.y_train = sparse.csr_matrix(self.y_train)
                            # self.y_test = sparse.csr_matrix(self.y_test)
                            # self.clf.fit(self.X_train, self.y_train)
                        finally:
                            pass
                    #
                    finally:
                        pass
                finally:
                    pass
            finally:
                pass
        finally:
            # pass
            try:
                self.clf.fit(self.X_train, self.y_train)
            except Exception as e:
                print(str(e) + "\n" + "Unable to evaluate")
                return "error"
            # else:
            #     print(style.GREEN +"Evaluated Successfully")
            #     print(style.BLACK)

        y_pred = self.clf.predict(self.X_test)

        ind_predictions_prob = []
        ind_predictions_pro = []
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test , y_pred)
        self.cfmatrix.append(cm)
        try:
            if self.split != "k-fold":
                ind_predictions_prob.extend(self.clf.predict_proba(self.X_train).toarray())
                ind_predictions_prob.extend(self.clf.predict_proba(self.X_test).toarray())
                ind_predictions_pro.extend(self.clf.predict_proba(self.X_test).toarray())
            else:
                ind_predictions_pro.extend(self.clf.predict_proba(self.X_test).toarray())
        except:
            try:
                if self.split != "k-fold":
                    ind_predictions_prob.extend(np.array(self.clf.predict_proba(self.X_train)))
                    ind_predictions_prob.extend(np.array(self.clf.predict_proba(self.X_test)))
                    ind_predictions_pro.extend(np.array(self.clf.predict_proba(self.X_test)))
                else:
                    ind_predictions_pro.extend(np.array(self.clf.predict_proba(self.X_test)))


            except Exception as e:
                print(str(e) + "\n" + "Unable to evaluate")
                # print(style.BLACK)

        max_probs = []
        for item in ind_predictions_prob:
            max_probs.append(item[0])
        max_probs = np.array(max_probs)
        acc, pre, rec, f1, spec, mcc, roc, sens, prc = Utils.fundamentalEvalMetrics(
            y_pred, self.y_test, ind_predictions_pro)
        print(acc, pre, rec, f1, spec, mcc, roc, sens, prc)
        self.averageScores["acc"] = self.averageScores["acc"] + acc
        self.averageScores["pre"] = self.averageScores["pre"] + pre
        self.averageScores["rec"] = self.averageScores["rec"] + rec
        self.averageScores["f1"] = self.averageScores['f1'] + f1
        self.averageScores["spec"] = self.averageScores['spec'] + spec
        self.averageScores["mcc"] = self.averageScores['mcc'] + mcc
        self.averageScores["sensitiviy"] = self.averageScores['sensitiviy'] + sens
        self.averageScores["AUROC"] = self.averageScores['AUROC'] + roc
        self.averageScores["AUPRC"] = self.averageScores['AUPRC'] + prc

        self.actual_test_reg_labels.extend(self.y_test.copy())
        self.predicted_test_reg_labels.extend(y_pred.copy())
        self.predicted_test_reg_prob.extend(ind_predictions_pro.copy())

        return "success"

    def evaluate_kfold_version1(self):
        self.re_init_paramas()

        df = pd.read_csv(self.encFeaturesPath)
        labels = df[df.columns[-1]]  # df.iloc[:,-1:]
        labels = np.array(labels.values.tolist()[:])

        tf_labels = labels

        df.drop(df.columns[-1], axis=1, inplace=True)

        features = np.array(df)

        kf = StratifiedKFold(n_splits=self.kfold, random_state=42, shuffle=True)
        fold = 0
        for train_index, test_index in kf.split(features, tf_labels):

            self.clf = self.reintiliazeclassifier(self.classifierfunc, self.classifierSaveName)

            self.X_train, self.X_test = features[train_index], features[test_index]
            self.y_train, self.y_test = np.array(labels)[train_index], np.array(labels)[test_index]

            try:
                res = self.classify()
                if res == "error":
                    return res
            except Exception as e:
                print( str(e) + "\n" + "Unable to evaluate")
                # print(style.BLACK)
                return "error"
            fold += 1

        self.averageScores["acc"] = self.averageScores["acc"] / self.kfold
        self.averageScores["pre"] = self.averageScores["pre"] / self.kfold
        self.averageScores["rec"] = self.averageScores["rec"] / self.kfold
        self.averageScores["f1"] = self.averageScores['f1'] / self.kfold
        self.averageScores["spec"] = self.averageScores["spec"] / self.kfold
        self.averageScores["mcc"] = self.averageScores["mcc"] / self.kfold
        self.averageScores["AUPRC"] = self.averageScores["AUPRC"] / self.kfold
        self.averageScores["sensitiviy"] = self.averageScores["sensitiviy"] / self.kfold
        self.averageScores["AUROC"] = self.averageScores["AUROC"] / self.kfold

        return res

        # "Results/Actual_Predicted_of_Algos"

    def evaluate_independent(self):


        self.re_init_paramas()

        df = pd.read_csv(self.encFeaturesPath)
        ds_type = df[df.columns[-1]]
        df.drop(df.columns[-1], axis=1, inplace=True)
        labels = df[df.columns[-1]]

        tf_labels=labels


        df.drop(df.columns[-1], axis=1, inplace=True)

        train_index = [i for i, x in enumerate(ds_type) if x == "train"]
        test_index = [i for i, x in enumerate(ds_type) if x == "test"]
        self.X_train, self.X_test = np.array(df)[train_index], np.array(df)[test_index]
        self.y_train, self.y_test = np.array(labels)[train_index], np.array(labels)[test_index]
        self.cm_class = set(tf_labels)
        self.df_add = pd.DataFrame(pd.DataFrame(index=range(len(self.cm_class)), columns=range(len(self.cm_class))))
        self.df_add.columns = self.cm_class
        self.df_add.index = self.cm_class
        self.clf = self.reintiliazeclassifier(self.classifierfunc, self.classifierSaveName)
        try:
            res = self.classify()
            if res == "error":
                return res
        except Exception as e:
            print(+ str(e) + "\n" + "Unable to evaluate")
            # print(style.BLACK)
            return "error"
        return res

    def saveActualAndPredicted(self, actual, pred, pred_prob, name, path):
        file = os.path.join(path, name + ".csv")
        writer = codecs.open(file, 'w')
        writer.write("actual" + "\t" + "predicted" + "\t" + "predicted_prob" + "\n")
        size = len(actual)
        if len(pred_prob) != 0:
            for actual_pre in range(size):
                writer.write(
                    str(actual[actual_pre]) + "\t" + str(pred[actual_pre]) + "\t" + str(pred_prob[actual_pre]) + "\n")

        else:
            for actual_pre in range(size):
                writer.write(str(actual[actual_pre]) + "\t" + str(pred[actual_pre]) + "\n")

    def saveClassfierResults(self, scores, filename):
        wrpath=self.savePath

        print(wrpath)
        scoresavePath=os.path.join(wrpath, filename)
        writer=codecs.open(scoresavePath, 'w')

        keys=scores.keys()
        for key in keys:
            writer.write(str(key)+"\t")

        writer.write("\n")
        for val in scores.values():
            writer.write(str(val)+"\t")

    def saveClassfierResultsAllEncoderSettings(self, fenames, scoresofallenc, filename):
        wrpath=self.savePath
        scoresavePath=os.path.join(wrpath, filename+".csv")

        res=pd.DataFrame(scoresofallenc)
        res.insert(0, "encoder",fenames)
        res.to_csv(scoresavePath, index=False, sep="\t")
        allresults = os.path.join(wrpath, "All_results.csv")
        if not os.path.exists(allresults):
            res.to_csv(allresults,index=False)
        else:
            df=pd.read_csv(allresults)
            final_df=pd.concat([df,res],ignore_index=False)
            final_df.drop_duplicates(keep="last", inplace=True)
            final_df.to_csv(allresults,index=False)

    def RunClassifier(self, classifierfunc=None, classifierSaveName=None):
        self.classifierfunc = classifierfunc
        self.classifierSaveName = classifierSaveName
        self.resultsofAllEncoders = []
        self.namesofAllEncoders = []
        self.cfmatrix = []

        print("start evaluation")
        files = [(files) for r, d, files in os.walk(self.encodedSequencesPath)][0]
        custom_files = []
        if self.custom_encoders_names != "all":
            for enc in self.custom_encoders_names:
                for file in files:
                    if enc in file.split("[")[0]:
                        if file not in custom_files:
                            custom_files.append(file)
        else:
            custom_files = files

        files = custom_files.copy()
        for encFeatures in files:
            print(encFeatures)
            self.encFeatures=encFeatures
            print("currently evaluating"+ str(self.encFeatures))
            self.encFeaturesPath = os.path.join(self.encodedSequencesPath, self.encFeatures)
            if self.split == "k-fold":
                res=self.evaluate_kfold_version1()
                if res == "error":
                    continue
                self.save_kfold_results()
            else:
                res=self.evaluate_independent()
                if res == "error":
                    continue
                self.save_ind_results()

        print(self.averageScores)
        if len(self.resultsofAllEncoders)>=1:
            self.saveClassfierResultsAllEncoderSettings(self.namesofAllEncoders, self.resultsofAllEncoders, self.classifierSaveName)
    def getsupportedClassifiers(self):
        allclassifiers = dict()
        from sklearn.ensemble import RandomForestClassifier

        allclassifiers["RandomForestClassifier"] = RandomForestClassifier(random_state=self.random_state)
        return allclassifiers


    def getCurrentClassifier(self, name):
        allreg=self.getsupportedClassifiers()
        return allreg[name]

    def getnamesofSupportedClassifiers(self):
        all=self.getsupportedClassifiers()
        return list(all.keys())

    def reintiliazeclassifier(self, clf_funct, clf_name):
        dashclassifier=False
        classifiers = self.getsupportedClassifiers()
        if"-" in clf_name:
            clf_name, cstom_classifier=clf_name.split("-")
            dashclassifier=True
        clf_funct=classifiers[clf_name]
        multiclass_config = Config(config_path=self.multiclass_config_file)
        for param in multiclass_config.data[clf_name].dict.items():
            if param[0] == "base_estimator":
                setattr(clf_funct, param[0], globals()[param[1]]())
            else:
                setattr(clf_funct, param[0], param[1])
        classifiers[clf_name] = clf_funct
        return clf_funct

    def update_params(self, classifiers):
        multiclass_config=Config(config_path=self.multiclass_config_file)
        for clf_name, clf_funct in classifiers.items():
            for param in multiclass_config.data[clf_name].dict.items():
                if param[0] == "base_estimator":
                    setattr(clf_funct, param[0], globals()[param[1]]())
                else:
                    setattr(clf_funct, param[0], param[1])
            classifiers[clf_name] = clf_funct

        return classifiers

    def evaluate_kfold_version2_optimal(self, features, labels):
        self.re_init_paramas()
        kf = StratifiedKFold(n_splits=self.kfold, random_state=42, shuffle=True)
        fold=0
        # if self.config.data["task_mode"] == "regression":
        #     tf_labels = self.data_transform(labels)
        # else:
        tf_labels = labels
        for train_index, test_index in kf.split(features, tf_labels):

            self.clf = self.reintiliazeclassifier(self.classifierfunc, self.classifierSaveName)

            self.X_train, self.X_test = np.array(features)[train_index], np.array(features)[test_index]
            self.y_train, self.y_test = np.array(labels)[train_index], np.array(labels)[test_index]
            try:
                res = self.classify()
                if res == "error":
                    return res
            except Exception as e:
                print(str(e) + "\n" + "Unable to evaluate")
                # print(style.BLACK)
                return "error"
            fold+=1


            self.averageScores["acc"] = self.averageScores["acc"] / self.kfold
            self.averageScores["pre"] = self.averageScores["pre"] / self.kfold
            self.averageScores["rec"] = self.averageScores["rec"] / self.kfold
            self.averageScores["f1"] = self.averageScores['f1'] / self.kfold
            self.averageScores["spec"] = self.averageScores["spec"] / self.kfold
            self.averageScores["mcc"] = self.averageScores["mcc"] / self.kfold
            self.averageScores["AUPRC"] = self.averageScores["AUPRC"] / self.kfold
            self.averageScores["sensitiviy"] = self.averageScores["sensitiviy"] / self.kfold
            self.averageScores["AUROC"] = self.averageScores["AUROC"] / self.kfold

    def saveClassifierResultsLevel2(self, scores, filename, path):

        filename=self.classifierSaveName+"-"+filename
        scoresavePath = os.path.join(path, "level2res.csv")
        headerwrite=True
        if os.path.exists(scoresavePath):
            headerwrite=False

        writer = codecs.open(scoresavePath, 'a')
        keys=scores.keys()
        if headerwrite==True:
            for key in keys:
                writer.write(str(key) + "\t")
            writer.write("\n")

        writer.write(filename+"\t")
        for val in scores.values():
            writer.write(str(val) + "\t")

        writer.write("\n")
        print("results have been saved for file=" + str(filename))

    def evaluate_independent_optimal(self,features, labels, ds_type):
        self.re_init_paramas()

        train_index = [i for i, x in enumerate(ds_type) if x == "train"]
        test_index = [i for i, x in enumerate(ds_type) if x == "test"]
        self.X_train, self.X_test =  np.array(features)[train_index], np.array(features)[test_index]
        self.y_train, self.y_test = np.array(labels)[train_index], np.array(labels)[test_index]
        self.clf = self.reintiliazeclassifier(self.classifierfunc, self.classifierSaveName)
        try:
            res = self.classify()
            if res == "error":
                return res
        except Exception as e:
            print( str(e) + "\n" + "Unable to evaluate")
            # print(style.BLACK)
            return "error"
        return res

    def ClassifierProvdingDataFrameAtSettingOptimalLevel(self,  clf,classifierSaveName, features,  labels,ds_type, fileName, path, previousbestperformance=None, fold=None):
        clf = self.reintiliazeclassifier(clf, classifierSaveName)
        self.classifierfunc = clf
        self.classifierSaveName = classifierSaveName
        self.actual_test_reg_labels = []
        self.predicted_test_reg_labels = []
        self.predicted_test_reg_prob = []
        if self.split == "k-fold":
            self.evaluate_kfold_version2_optimal(features, labels)
            if previousbestperformance == None:
                self.saveClassifierResultsLevel2(self.averageScores, str(self.kfold) + "-" + fileName + "-res.csv",
                                                 path)
            else:
                if self.averageScores["acc"] > previousbestperformance:
                    self.saveClassifierResultsLevel2(self.averageScores,
                                                     str(self.kfold) + "-" + fileName + "-res.csv",
                                                     path)

        else:
            self.evaluate_independent_optimal(features, labels, ds_type)
            if previousbestperformance == None:
                self.saveClassifierResultsLevel2(self.averageScores, str(self.split) + "-" + fileName + "-res.csv", path)
            else:
                if self.averageScores["acc"] > previousbestperformance:
                    self.saveClassifierResultsLevel2(self.averageScores, str(self.split) + "-" + fileName + "-res.csv",
                                                     path)


        return self.averageScores


    def Level_1_Evaluation(self):
        classifiers = dict()

        if self.config.data["Evaluation"]["classifiers"]=="all":
            classifiers = self.getsupportedClassifiers()
        else:
            for clf_names in self.config.data["Evaluation"]["classifiers"]:
                classifiers[clf_names]=self.getCurrentClassifier(clf_names)

        classifiers=self.update_params(classifiers)

        self.encodedSequencesPath = self.encoded_enc_SequencesPath
        print(self.encodedSequencesPath)
        print(classifiers)
        [self.RunClassifier(
                            classifierfunc=clf_func, classifierSaveName=clf) for clf, clf_func in
         classifiers.items()]
        print("done with level 1 evaluation")
