package feature_extraction;

import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import libsvm.svm_node;
import libsvm.svm_problem;
import misc.Config;
import misc.DummyPreprocessor;
import misc.ParagraphIDAwareIterator;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;


/**
 * Created by sotos on 4/30/16.
 */
public class Paragraph2Vec {
    Config config;
    HashMap<String, ArrayList<Integer>> parIdToParVector; //a map from paragraph IDs to paragraph vectors
    String parPrefix = "paragraphID_";
    private InMemoryLookupTable<VocabWord> lookupTable;
    private HashMap<Integer, HashSet<Instance>> wordIDToDocuments = null;


    public Paragraph2Vec(Config config) {
        this.config = config;
    }

    public Paragraph2Vec(Config config, InMemoryLookupTable lookupTable) {
        this.config = config;
        this.lookupTable = lookupTable;

    }

    /**
     * Trains paragraph vectors
     * @param instances
     */
    public void trainModel(InstanceList instances) {
        ParagraphIDAwareIterator iterator = new ParagraphIDAwareIterator(instances);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new DummyPreprocessor());

        //read parameters from config file
        double learningRate = Double.parseDouble(config.getProperty("w2v.learningRate", "0.025"));
        double minLearningRate = Double.parseDouble(config.getProperty("w2v.MinLearningRate", "0.000001"));
        int batchSize = Integer.parseInt(config.getProperty("w2v.batchSize", "1000"));
        int epochs = Integer.parseInt(config.getProperty("w2v.epochs", "1000"));
        boolean trainWordVectors = Boolean.parseBoolean(config.getProperty("w2v.trainWordVectors", "false"));
        int layerSize = Integer.parseInt(config.getProperty("w2v.layerSize", "100"));
        int windowSize = Integer.parseInt(config.getProperty("w2v.windowSize", "5"));
        int minWordFrequency = Integer.parseInt(config.getProperty("w2v.minWordFrequency", "1"));
        double subSampling = Double.parseDouble(config.getProperty("w2v.subSampling", "0"));
        double negSampling = Double.parseDouble(config.getProperty("w2v.negSampling", "0"));
        //iterations = number of iterations done for each mini-batch during training
        //useAdaGrad This method defines whether adaptive gradients should be used or not

        // ParagraphVectors training configuration
        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                .learningRate(learningRate)
                .minLearningRate(minLearningRate)
                .batchSize(batchSize)
                .epochs(epochs)
                .trainWordVectors(trainWordVectors)
                .iterate(iterator)
                .layerSize(layerSize)
                .windowSize(windowSize)
                .tokenizerFactory(t)
                .minWordFrequency(minWordFrequency)
                .sampling(subSampling)
                .negativeSample(negSampling)
                .seed(1234)
                .build();

        paragraphVectors.fit();

        //paragraphVectors.getLookupTable().vector("paragraphID_37879");
        lookupTable = (InMemoryLookupTable) paragraphVectors.getLookupTable();
    }

    /**
     * Transforms instances to svm_problem based on the trained paragraph vectors.
     * @param instances
     * @return
     */
    public svm_problem toSparseMatrix(InstanceList instances) {
        svm_problem problem = new svm_problem();
        problem.l = instances.size();
        problem.y = new double[problem.l];
        problem.x = new svm_node[problem.l][];
        int i = 0;
        for (i = 0; i < instances.size(); i++) { //each instance corresponds to a document
            Instance inst = instances.get(i);
            String paragraphID = (String) inst.getName();
            paragraphID = parPrefix + paragraphID;

            INDArray vector = lookupTable.vector(paragraphID);
            svm_node[] x = new svm_node[vector.size(1)];
            //indeces sequencial
            NdIndexIterator iter = new NdIndexIterator(1, vector.size(1));
            int j = 0;
            while (iter.hasNext()) {
                int[] nextIndex = iter.next();
                double nextVal = vector.getDouble(nextIndex);
                x[j] = new svm_node();
                x[j].index = j + 1;
                x[j].value = nextVal;
                j++;
            }
            problem.y[i] = Double.parseDouble((String) inst.getTarget());
            problem.x[i] = x;
        }
        return problem;
    }

    /**
     * Same as toSparseMatrix, but adds BOW features to the svm_problem
     * @param instances
     * @return
     */
    public svm_problem toSparseMatrixPlusBow(InstanceList instances) {
        svm_problem problem = new svm_problem();
        problem.l = instances.size();
        problem.y = new double[problem.l];
        problem.x = new svm_node[problem.l][];

        if (wordIDToDocuments == null) {
            wordIDToDocuments = new HashMap<Integer, HashSet<Instance>>();
        }

        for (int i = 0; i < instances.size(); i++) { //one instance is one document
            Instance inst = instances.get(i);
            FeatureVector tokens = (FeatureVector) inst.getData();
            int[] indices = tokens.getIndices();
            double[] values = tokens.getValues();
            HashMap<Integer, Integer> features = new HashMap<Integer, Integer>(); //One for every document. map from cluster to times of occurences for an instance
            for (int k = 0; k < indices.length; k++) { //for every word in the document
                if (!wordIDToDocuments.containsKey(indices[k])) {
                    wordIDToDocuments.put(indices[k], new HashSet<Instance>());
                }
                wordIDToDocuments.get(indices[k]).add(inst);
            }
        }


        for (int i = 0; i < instances.size(); i++) { //each instance corresponds to a document
            Instance inst = instances.get(i);
            FeatureVector tokens = (FeatureVector) inst.getData();
            int[] indices = tokens.getIndices();
            double[] values = tokens.getValues();

            //add paragraph vectors
            String paragraphID = (String) inst.getName();
            paragraphID = parPrefix + paragraphID;
            INDArray vector = lookupTable.vector(paragraphID);
            svm_node[] x = new svm_node[vector.size(1)+indices.length];
            //indeces sequencial
            NdIndexIterator iter = new NdIndexIterator(1, vector.size(1));
            int j = 0;
            while (iter.hasNext()) {
                int[] nextIndex = iter.next();
                double nextVal = vector.getDouble(nextIndex);
                x[j] = new svm_node();
                x[j].index = j + 1;
                x[j].value = nextVal;
                j++;
            }


            //add bow features
            for (int k = vector.size(1); k < vector.size(1)+indices.length; k++) {
                double w_tf = 1 + Math.log10((double) values[k - vector.size(1)]);
                double N = (double) instances.size();
                double df = (double) wordIDToDocuments.get(indices[k - vector.size(1)]).size();
                double w_idf = Math.log10(N / df);
                double w = w_tf * w_idf;
                x[k] = new svm_node();
                x[k].index = indices[k - vector.size(1)];
                x[k].value = w;
            }

            problem.y[i] = Double.parseDouble((String) inst.getTarget());
            problem.x[i] = x;
        }
        return problem;
    }

    /**
     * Saves lookupTable, which contains the paragraph vectors to a file
     */
    public void saveVectors() {
        try {
            //serialize problem object
            FileOutputStream fileOut = new FileOutputStream(Config.returnOutputNamePrefix(config)
                    + ".vectors");
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(lookupTable);
            out.close();
            fileOut.close();
            System.out.println("finished Serializing word/paragraph vectors");
        } catch (Exception e) {
            System.err.println(e.getMessage());
            System.err.println(e.getStackTrace());
            System.err.println(e.getCause());
        }
    }


}

