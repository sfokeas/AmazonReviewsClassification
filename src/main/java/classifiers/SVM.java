package classifiers;

import libsvm.*;
import misc.Config;

import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;

/**
 * Created by sotos on 3/9/16.
 */
public class SVM {
    Config config;
    svm_problem trainProb; //one of these can be null
    svm_problem testProb;
    svm_parameter param;

    public SVM(Config conf, svm_problem trainProb, svm_problem testProb) {
        config = conf;
        this.trainProb = trainProb;
        this.testProb = testProb;
        param = new svm_parameter();
    }

    public SVM(Config conf) {
        config = conf;
        param = new svm_parameter();
        testProb = new svm_problem();
    }

    /**
     * Produces statistics out of a predictions file
     * @throws IOException
     */
    public void predFileToMeasurements() throws IOException {
        BufferedReader predFile = new BufferedReader(new FileReader(config.getProperty("data.predictionsFile")));
        ArrayList<Double> arrayY = new ArrayList<Double>();
        ArrayList<Double> arrayPred = new ArrayList<Double>();
        String line;
        line = predFile.readLine(); //ignore first line = header
        while ((line = predFile.readLine()) != null) {
            String[] lineFields = line.split("[\\t]");
            arrayY.add(Double.parseDouble(lineFields[0]));
            arrayPred.add(Double.parseDouble(lineFields[1]));
        }
        predFile.close();
        testProb.l = arrayPred.size();
        testProb.y = new double[testProb.l];
        double[] predicted = new double[testProb.l];
        for (int i = 0; i < testProb.l; i++) {
            testProb.y[i] = arrayY.get(i);
            predicted[i] = arrayPred.get(i);
        }
        writeResults(predicted, testProb, "results");
    }


    private void saveModel(svm_model model) {
        try {
            String outputFilename = Config.returnOutputNamePrefix(config) + ".model";
            svm.svm_save_model(outputFilename, model);
        } catch (IOException e) {
            System.err.println(e.fillInStackTrace());
            System.err.println(e.getMessage());
        }
    }

    /**
     * Write performance statistics
     * @param predicted
     * @param prob
     * @param resultsTitle
     */
    private void writeResults(double[] predicted, svm_problem prob, String resultsTitle) {
        try {
            BufferedWriter rawResultsWriter = new BufferedWriter(new FileWriter(Config.returnOutputNamePrefix(config) + ".predictedValues", true));
            BufferedWriter measurementsWriter = new BufferedWriter(new FileWriter(Config.returnOutputNamePrefix(config) + ".statistics", true));
            measurementsWriter.write("\n" + resultsTitle + "\n");
            rawResultsWriter.write("\n" + resultsTitle + "\n");
            measurementsWriter.write(config.toString() + "\n");
            rawResultsWriter.write(config.toString() + "\n");
            //calculate and write quality of predictions measurements
            ArrayList<Double> classes = findClasess(prob); //TODO if number of classes in config file is not set
            double[][] confusionMatrix = new double[classes.size()][classes.size()]; //rows are the predictions, columns are true values
            //initialize confusionMatrix with zero
            for (int i = 0; i < classes.size(); i++) {
                for (int j = 0; j < classes.size(); j++) {
                    confusionMatrix[i][j] = 0;
                }
            }
            for (int i = 0; i < prob.l; i++) {
                confusionMatrix[classes.indexOf(predicted[i])][classes.indexOf(prob.y[i])]++;
            }

            //calculate precision, recall and F-score, BAC for in class X / not in class X binary decisions.
            double sumPrec = 0;
            double sumRecall = 0;
            NumberFormat formatter = new DecimalFormat("#.###");
            for (int c = 0; c < classes.size(); c++) {
                double truePos = confusionMatrix[c][c]; //true positives
                double falsePos = 0, falseNeg = 0;
                for (int i = 0; i < classes.size(); i++) {
                    if (i == c) continue;
                    falsePos += confusionMatrix[c][i];
                    falseNeg += confusionMatrix[i][c];
                }
                double trueNeg = prob.l - truePos - falsePos - falseNeg; //tn = total - tp - fp - fn
                double prec = truePos / (truePos + falsePos);
                double recall = truePos / (truePos + falseNeg);
                double fscore = (2 * prec * recall) / (prec + recall);
                double specificity = trueNeg / (trueNeg + falsePos);
                double sensitivity = truePos / (truePos + falseNeg);
                double bac = (specificity + sensitivity) / 2;

                measurementsWriter.write("Class " + classes.get(c) + "\t"
                        + "precision: " + formatter.format(prec) + "\t"
                        + "recall: " + formatter.format(recall) + "\t"
                        + "F1-Score: " + formatter.format(fscore) + "\t"
                        + "specificity: " + formatter.format(specificity) + "\t"
                        + "sensitivity: " + formatter.format(sensitivity) + "\t"
                        + "BAC: " + formatter.format(bac) + "\n"
                );
                sumPrec += prec;
                sumRecall += recall;
            }
//            precision
//                    Precision = true_positive / (true_positive + false_positive)
//            recall
//
//                    Recall = true_positive / (true_positive + false_negative)
//
//            fscore
//
//            F-score = 2 * Precision * Recall / (Precision + Recall)
//
//            bac
//
//            BAC (Balanced ACcuracy) = (Sensitivity + Specificity) / 2,
//                    where Sensitivity = true_positive / (true_positive + false_negative)
//            and   Specificity = true_negative / (true_negative + false_positive)
//

            //calculate accuracy
            int total_correct = 0;
            for (int i = 0; i < prob.l; i++)
                if (predicted[i] == prob.y[i])
                    ++total_correct;
            double avePrec = sumPrec / classes.size();
            double aveRecall = sumRecall / classes.size();
            double macroFScore = (2 * avePrec * aveRecall) / (avePrec + aveRecall);
            measurementsWriter.write("General Accuracy = " + 100.0 * total_correct / prob.l + "%\t"
                    + "Average Precision = " + formatter.format(avePrec) + "\t"
                    + "Average Recall = " + formatter.format(aveRecall) + "\t"
                    + "Macro F1-Score = " + formatter.format(macroFScore) + "\n"
            );

            //write predicted values
            rawResultsWriter.write("real\tpredicted\n");
            for (int i = 0; i < predicted.length; i++) {
                rawResultsWriter.write(Double.toString(prob.y[i]) + "\t" + Double.toString(predicted[i]) + "\n");
            }
            rawResultsWriter.flush();
            rawResultsWriter.close();
            measurementsWriter.flush();
            measurementsWriter.close();
        } catch (IOException e) {
            System.err.println(e.fillInStackTrace());
            System.err.println(e.getMessage());
        }
    }

    /**
     * Does cross validation over the training set
     * @param nr_folds (numbeer of folds)
     * @return
     */
    private void do_cross_validation(int nr_folds) {
        int noElementPerFold = (int) trainProb.l / nr_folds; //we might miss some elements. less than nr_folds, which is small.
        ArrayList<Integer> shuffledList = new ArrayList<Integer>();
        for (int i = 0; i < noElementPerFold * nr_folds; i++) {
            shuffledList.add(i);
        }
        // Shuffle it to produce random sets
        Collections.shuffle(shuffledList);
        double[] finalTarget = new double[trainProb.l];
        for (int fold = 0; fold < nr_folds; fold++) {
            int valSetBegin = fold * noElementPerFold;
            int valSetEnd = fold * noElementPerFold + noElementPerFold - 1;
            svm_problem subProb = new svm_problem();
            subProb.l = noElementPerFold * (nr_folds - 1);
            subProb.x = new svm_node[subProb.l][];
            subProb.y = new double[subProb.l];
            for (int i = 0; i < valSetBegin; i++) {
                subProb.x[i] = trainProb.x[shuffledList.get(i)];
                subProb.y[i] = trainProb.y[shuffledList.get(i)];
            }
            for (int i = valSetEnd + 1; i < nr_folds * noElementPerFold; i++) {
                subProb.x[i - noElementPerFold] = trainProb.x[shuffledList.get(i)];
                subProb.y[i - noElementPerFold] = trainProb.y[shuffledList.get(i)];
            }
            svm_model submodel = svm.svm_train(subProb, param);


            //training errors for fold
            double[] trainTarget = new double[subProb.l];
            for (int i = 0; i < subProb.l; i++) {
                trainTarget[i] = svm.svm_predict(submodel, subProb.x[i]);
            }
            writeResults(trainTarget, subProb, "TRAINING ERRORS FOLD " + Integer.toString(fold + 1));

            //validation set errors for fold
            svm_problem testSubProb = new svm_problem();
            testSubProb.l = noElementPerFold;
            testSubProb.x = new svm_node[testSubProb.l][];
            testSubProb.y = new double[testSubProb.l];
            double[] testTarget = new double[testSubProb.l];

            for (int i = valSetBegin; i <= valSetEnd; i++) {
                testSubProb.x[i - valSetBegin] = trainProb.x[shuffledList.get(i)];
                testSubProb.y[i - valSetBegin] = trainProb.y[shuffledList.get(i)];
                testTarget[i - valSetBegin] = svm.svm_predict(submodel, testSubProb.x[i - valSetBegin]);
                finalTarget[shuffledList.get(i)] = testTarget[i - valSetBegin];
            }
            writeResults(testTarget, testSubProb, "VALIDATION ERRORS FOLD " + Integer.toString(fold + 1));
            System.out.println("Fold " + Integer.toString(fold + 1) + " finished");
        }
        //accumulate results for final validation set errors
        writeResults(finalTarget, trainProb, "CUMULATIVE VALIDATION ERRORS ALL FOLDS");
    }


    private ArrayList<Double> findClasess(svm_problem prob) {
        ArrayList<Double> classes = new ArrayList<Double>();
        for (int i = 0; i < prob.l; i++) {
            if (!classes.contains(prob.y[i])) {
                classes.add(prob.y[i]);
            }
        }
        return classes;
    }

    /**
     * Main entry point. Decides which action should be taken.
     */
    public void runSVM() {

        ///svm train parameters
        //default parameters. Most of them not used or don't change. They only parameter that changes is C.
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.LINEAR;
        param.degree = 3;
        param.gamma = 0;
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 40;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = new int[0];
        param.weight = new double[0];
        //get rest parameters from config file
        param.C = Double.parseDouble(config.getProperty("clf.svm.C", "1"));
        //---

        svm_model model = null;
        // cross validation or train or predict
        if (config.getProperty("clf.crossValidation").trim().equalsIgnoreCase("true")) {
            do_cross_validation(Integer.parseInt(config.getProperty("clf.crossValidation.folds")));
        } else if (config.getProperty("clf.train").trim().equalsIgnoreCase("true")) {
            model = svm.svm_train(trainProb, param);
            saveModel(model);
            if (config.getProperty("clf.predict").trim().equalsIgnoreCase("true")) { //covers the train-predict combo
                double[] target = new double[testProb.l];
                for (int i = 0; i < testProb.l; i++) {
                    target[i] = svm.svm_predict(model, testProb.x[i]); //predY just a scalar not a vector of values. Corresponds only to one feature vector
                }
                writeResults(target, testProb, "predictions");
            }
        } else if (config.getProperty("clf.predict").trim().equalsIgnoreCase("true")) { //only predict. load model from file
            try {
                model = svm.svm_load_model(config.getProperty("clf.predict.model").trim());

                double[] target = new double[testProb.l];
                for (int i = 0; i < testProb.l; i++) {
                    target[i] = svm.svm_predict(model, testProb.x[i]); //predY just a scalar not a vector of values. Corresponds only to one feature vector
                }
                writeResults(target, testProb, "predictions");
            } catch (IOException e) {
                System.err.println(e.fillInStackTrace());
                System.err.println(e.getMessage());
            }
        }
    }

    private class scaleMeasures implements java.io.Serializable {
        public double max;
    }

}
