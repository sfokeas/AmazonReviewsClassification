import cc.mallet.types.Alphabet;
import cc.mallet.types.InstanceList;
import classifiers.SVM;
import feature_extraction.BOW;
import feature_extraction.BrownClustering;
import feature_extraction.Paragraph2Vec;
import libsvm.svm_problem;
import misc.Config;
import misc.DataImporter;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;


/**
 * Created by sotos on 5/3/16.
 */
public class TextClassification {
    private static TextClassification ourInstance = new TextClassification();

    private TextClassification() {
    }

    public static TextClassification getInstance() {
        return ourInstance;
    }

    public static void main(String[] args) {
        try {
            //parse config file
            //config has some defaults, but they can be overridden by args
            final long startTime = System.currentTimeMillis();
            svm_problem trainProblem = null;
            svm_problem testProblem = null;
            Config config = new Config(args[0]);
            //parse arguments
            // currently: application <config file location> <C> //C is the cost parameter of SVM
            if (args.length > 1) {
                config.setProperty("clf.svm.C", args[1]);
                if (args.length > 2) {
                    if (args.length == 7) {
                        //application <config file location> <C> <learningRate> <batchSize> <epochs> <layerSize> <windowSize>
                        config.setProperty("w2v.learningRate", args[2]);
                        config.setProperty("w2v.batchSize", args[3]);
                        config.setProperty("w2v.epochs", args[4]);
                        config.setProperty("w2v.layerSize", args[5]);
                        config.setProperty("w2v.windowSize", args[6]);
                    } else {
                        System.err.println("usage: application <config file location> <C> <learningRate> <batchSize> <epochs> <layerSize> <windowSize>");
                        return;
                    }
                }
            }
            //config.list(System.out);
            System.out.println(config.toString());
            //Only one of the following four inputs are used each run. The priority goes like this: predicted > problem > instances > input.
            if (!config.getProperty("data.predictionsFile").trim().equalsIgnoreCase("")) {
                //take a file with predictions and extract measurements
                SVM svm = new SVM(config);
                svm.predFileToMeasurements();
            } else { //if the predictions file is not available use the the serialized problem object
                if (!config.getProperty("data.problem").trim().equalsIgnoreCase("")) {
                    if (config.getProperty("clf.predict").trim().equalsIgnoreCase("true")) {
                        if (config.getProperty("data.test.problem").trim().equalsIgnoreCase("")) {
                            System.err.println("The same level of input should be provided for the test set as well!");
                            return;
                        }
                        FileInputStream fileIn = new FileInputStream(config.getProperty("data.test.problem"));
                        ObjectInputStream in = new ObjectInputStream(fileIn);
                        testProblem = (svm_problem) in.readObject();
                        in.close();
                        fileIn.close();
                    }
                    FileInputStream fileIn = new FileInputStream(config.getProperty("data.problem"));
                    ObjectInputStream in = new ObjectInputStream(fileIn);
                    trainProblem = (svm_problem) in.readObject();
                    in.close();
                    fileIn.close();
                    if (!config.getProperty("clf.predict").trim().equalsIgnoreCase("true")) {
                        testProblem = trainProblem;
                    }
                } else { //if the problem object is not available use the serialized instances object
                    DataImporter importer = new DataImporter(config);
                    InstanceList trainInstances = null;
                    InstanceList testInstances = null;
                    if (!config.getProperty("data.instances").trim().equalsIgnoreCase("")) {
                        if (config.getProperty("clf.predict").trim().equalsIgnoreCase("true")) {
                            if (config.getProperty("data.test.instances").trim().equalsIgnoreCase("")) {
                                System.err.println("The same level of input should be provided for the test set as well!");
                                return;
                            }
                            FileInputStream fileIn = new FileInputStream(config.getProperty("data.test.instances"));
                            ObjectInputStream in = new ObjectInputStream(fileIn);
                            testInstances = (InstanceList) in.readObject();
                            in.close();
                            fileIn.close();
                        }
                        FileInputStream fileIn = new FileInputStream(config.getProperty("data.instances"));
                        ObjectInputStream in = new ObjectInputStream(fileIn);
                        trainInstances = (InstanceList) in.readObject();
                        in.close();
                        fileIn.close();
                        if (!config.getProperty("clf.predict").trim().equalsIgnoreCase("true")) {
                            testInstances = trainInstances.subList(0,1);
                        }
                    } else { //if the serialized instances are not available extract them from input file
                        Alphabet alphabet = new Alphabet();
                        trainInstances = importer.importPreprocessed(alphabet, new File(config.getProperty("data.input")));
                        if (config.getProperty("clf.predict").trim().equalsIgnoreCase("true")) {
                            testInstances = importer.importPreprocessed(alphabet, new File(config.getProperty("data.test.input")));
                        }
                        else{
                            testInstances = trainInstances.subList(0,1);
                        }
                        //serialize the InstanceList for later reference
                        if (config.getProperty("data.saveInstances").trim().equalsIgnoreCase("true")) {
                            Path p = Paths.get(config.getProperty("data.input"));
                            String inputFilename = p.getFileName().toString();
                            //train instances
                            FileOutputStream trainFileOut = new FileOutputStream(config.getProperty("general.outputDir") + "/" + inputFilename + "_instances.ser");
                            ObjectOutputStream trainOut = new ObjectOutputStream(trainFileOut);
                            trainOut.writeObject(trainInstances);
                            trainOut.close();
                            trainFileOut.close();
                            //test instances
                            FileOutputStream testFileOut = new FileOutputStream(config.getProperty("general.outputDir") + "/" + inputFilename + "_instances.ser");
                            ObjectOutputStream testOut = new ObjectOutputStream(testFileOut);
                            testOut.writeObject(testInstances);
                            testOut.close();
                            testFileOut.close();
                            System.out.println("finished Serializing instances");
                        }
                    }
                    if (config.getProperty("data.model").trim().equals("bow")) {
                        BOW bow = new BOW(config);
                        if (config.getProperty("bow.type").trim().equals("binary")) {
                            trainProblem = bow.toSparseMatrixBinary(trainInstances);
                            testProblem = bow.toSparseMatrixBinary(testInstances);
                        } else {
                            trainProblem = bow.toSparseMatrix(trainInstances);
                            testProblem = bow.toSparseMatrix(testInstances);
                        }
                        System.out.println("finished transforming features from bow to sparse matrix");
                    } else if (config.getProperty("data.model").trim().equals("brown")) {
                        BrownClustering brown = new BrownClustering(config, testInstances.getDataAlphabet()); //train and test are supposed to have the same alphabet
                        if (config.getProperty("brown.plus_bow").trim().equalsIgnoreCase("true")) {
                            if (config.getProperty("brown.tfidf").trim().equalsIgnoreCase("true")) {
                                trainProblem = brown.toSparseMatrixIncludeBowTfidf(trainInstances);
                                testProblem = brown.toSparseMatrixIncludeBowTfidf(testInstances);
                            } else {
                                trainProblem = brown.toSparseMatrixIncludeBOW(trainInstances);
                                testProblem = brown.toSparseMatrixIncludeBOW(testInstances);
                            }
                        } else {
                            if (config.getProperty("brown.tfidf").trim().equalsIgnoreCase("true")) {
                                trainProblem = brown.toSparseMatrixTfidf(trainInstances);
                                testProblem = brown.toSparseMatrixTfidf(testInstances);
                            } else {
                                trainProblem = brown.toSparseMatrix(trainInstances);
                                testProblem = brown.toSparseMatrix(testInstances);
                            }
                        }
                        System.out.println("finished transforming features from brown to sparse matrix");
                    } else if (config.getProperty("data.model").trim().equals("word2vec")) {
                        Paragraph2Vec p2v = null;
                        if (config.getProperty("w2v.lookupTable", "").trim().equalsIgnoreCase("")) {
                            InstanceList p2vInstances = trainInstances.cloneEmpty();
                            p2vInstances.addAll(trainInstances);
                            p2vInstances.addAll(testInstances);
                            //p2vInstances = importer.importPreprocessed(new Alphabet(), new File(config.getProperty("w2v.corpus")));
                            p2v = new Paragraph2Vec(config);
                            //trainInstances.addAll(testInstances);
                            p2v.trainModel(p2vInstances); //p2vInstances should contain both train and test set
                            //trainInstances.removeAll(testInstances);
                            p2v.saveVectors();
                        } else { //else we have the pre-trained paragraph vectors
                            FileInputStream fileIn = new FileInputStream(config.getProperty("w2v.lookupTable"));
                            ObjectInputStream in = new ObjectInputStream(fileIn);
                            InMemoryLookupTable lookupTable = (InMemoryLookupTable) in.readObject();
                            in.close();
                            fileIn.close();
                            p2v = new Paragraph2Vec(config, lookupTable);
                        }
                        if (config.getProperty("w2v.plus_bow", "false").trim().equalsIgnoreCase("true")) {
                            trainProblem = p2v.toSparseMatrixPlusBow(trainInstances);
                            testProblem = p2v.toSparseMatrixPlusBow(testInstances);
                        } else {
                            trainProblem = p2v.toSparseMatrix(trainInstances);
                            testProblem = p2v.toSparseMatrix(testInstances);
                        }
                        System.out.println("finished extracting paragraph vectors");
                    }
                    if (trainInstances != null) {
                        System.out.println("Training set contains " + Integer.toString(trainInstances.getDataAlphabet().size()) + " different words");
                        if (testInstances != null) {
                            trainInstances.addAll(testInstances);
                            System.out.println("Both sets contains " + Integer.toString(trainInstances.getDataAlphabet().size()) + " different words");
                        }
                    }
                    if (testInstances != null) {
                        System.out.println("Test set contains " + Integer.toString(testInstances.getDataAlphabet().size()) + " different words");
                        if (trainInstances != null) {
                            testInstances.addAll(trainInstances);
                            System.out.println("Both sets contains " + Integer.toString(testInstances.getDataAlphabet().size()) + " different words");
                        }
                    }
                }
                if (trainProblem != null) {
                    System.out.println("Training problem contains " + Integer.toString(trainProblem.l) + " samples");
                }
                if (testProblem != null) {
                    System.out.println("Test problem contains " + Integer.toString(testProblem.l) + " samples");
                }
                SVM svm = new SVM(config, trainProblem, testProblem);//one of them can be null
                svm.runSVM(); //saves output to the specified file in config
                final long endTime = System.currentTimeMillis();
                System.out.println("Total execution time: " + (endTime - startTime));
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            System.err.println(e.getStackTrace());
            System.err.println(e.getCause());
        }
    }
}

