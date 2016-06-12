package feature_extraction;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import libsvm.svm_node;
import libsvm.svm_problem;
import misc.Config;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

/**
 * Created by sotos on 3/9/16.
 */
public class BrownClustering {
    private HashMap<Integer, HashSet<Integer>> wordIDsToFeatureIndices; //a map from words ids to features
    private HashMap<String, Integer> pathToUniqueFeatureIndex; //unique index for every path

    Config config;

    public BrownClustering(Config conf, Alphabet alphabet) throws IOException {
        config = conf;
        wordIDsToFeatureIndices = new HashMap<Integer, HashSet<Integer>>();
        pathToUniqueFeatureIndex = new HashMap<String, Integer>();
        if (config.getProperty("brown.only_clusters").trim().equalsIgnoreCase("true")) {
            loadClustersOnly(alphabet);
        } else {
            loadClusterPaths(alphabet); //TODO serialize mapIDsToCluster so you don't have to fill it up all the time. not a bottleneck
        }
    }

    /**
     * loads a file which contains the paths of the words as produced by wcluster and
     * inserts them into the into mapIDsToClusters.
     *
     * @param alphabet
     * @throws IOException
     */
    private void loadClusterPaths(Alphabet alphabet) throws IOException {
        //open file
        BufferedReader clustersFile = new BufferedReader(new FileReader(config.getProperty("data.brown.paths")));
        String line;
        int featureIndex = 1;
        while ((line = clustersFile.readLine()) != null) {
            String[] lineFields = line.split("[\\s]");
            String word = lineFields[1];
            String cluster = lineFields[0]; //binary representation of the path
            int cutoff = 0;
            if (!config.getProperty("brown.cutoff").equals("")) {
                cutoff = Integer.parseInt(config.getProperty("brown.cutoff")); //if cutoff is 0 then we consider all paths until the root. If not with cut off the last transitions from leafs to the root
            }
            for (int i = cluster.length(); i > cutoff; i--) {
                String subcluster = cluster.substring(0, i);
                //push the integer representation of the the cluster into the map
                Integer wordID = alphabet.lookupIndex(word); //	 Returns -1 if entry isn't present in the instances.
                if (wordID != -1) {
                    if (!pathToUniqueFeatureIndex.containsKey(subcluster)) {
                        pathToUniqueFeatureIndex.put(subcluster, featureIndex);
                        featureIndex++;
                    }
                    if (!wordIDsToFeatureIndices.containsKey(wordID)) {
                        wordIDsToFeatureIndices.put(wordID, new HashSet<Integer>());
                    }
                    wordIDsToFeatureIndices.get(wordID).add(pathToUniqueFeatureIndex.get(subcluster)); //parse the binary representation(2) as int.//use this integer as as the index of the feature. Sparse matrix so it don't matter if there are gaps
                }
            }
        }
    }



    /**
     * Only leaf clusters. Does not consider hierarchical paths
     *
     * @param alphabet
     * @throws IOException
     */
    private void loadClustersOnly(Alphabet alphabet) throws IOException {
        //open file
        BufferedReader clustersFile = new BufferedReader(new FileReader(config.getProperty("data.brown.paths")));
        String line;
        int featureIndex = 1;
        while ((line = clustersFile.readLine()) != null) {
            String[] lineFields = line.split("[\\s]");
            String word = lineFields[1];
            String cluster = lineFields[0]; //binary representation of the path
            if (!pathToUniqueFeatureIndex.containsKey(cluster)) {
                pathToUniqueFeatureIndex.put(cluster, featureIndex);
                featureIndex++;
            }
            Integer wordID = alphabet.lookupIndex(word); //	 Returns -1 if entry isn't present in the instances.
            if (wordID != -1) {
                if (!wordIDsToFeatureIndices.containsKey(wordID)) {
                    wordIDsToFeatureIndices.put(wordID, new HashSet<Integer>());
                }
                wordIDsToFeatureIndices.get(wordID).add(pathToUniqueFeatureIndex.get(cluster)); //parse the binary representation(2) as int.//use this integer as as the index of the feature. Sparse matrix so it don't matter if there are gaps
            }
        }
    }


    /**
     * Saves instances into a format which libSVM can easily handle
     *
     * @param instances
     * @return
     * @throws IOException
     */
    public svm_problem toSparseMatrix(InstanceList instances) throws IOException {
        svm_problem problem = new svm_problem();
        problem.l = instances.size();
        problem.y = new double[problem.l];
        problem.x = new svm_node[problem.l][];
        for (int i = 0; i < instances.size(); i++) { //one instance is one document
            Instance inst = instances.get(i);
            FeatureVector tokens = (FeatureVector) inst.getData();
            int[] indices = tokens.getIndices();
            double[] values = tokens.getValues();
            HashMap<Integer, Integer> features = new HashMap<Integer, Integer>(); //One for every document. map from cluster to times of occurences for an instance
            for (int k = 0; k < indices.length; k++) {
                if (wordIDsToFeatureIndices.containsKey(indices[k])) {
                    //Integer[] clustersList = wordIDsToFeatureIndices.get(indices[k]).toArray();
                    for (Integer cluster : wordIDsToFeatureIndices.get(indices[k])) {
                        if (!features.containsKey(cluster)) {
                            features.put(cluster, (int) values[k]);
                        } else {
                            features.put(cluster, features.get(cluster) + (int) values[k]);
                        }
                    }
                }
            }
            svm_node[] x = new svm_node[features.size()];
            int k = 0;
            for (Map.Entry<Integer, Integer> entry : features.entrySet()) {
                x[k] = new svm_node();
                x[k].index = entry.getKey(); //cluster
                x[k].value = entry.getValue(); //number of occurrences of this cluster.
                k++;
            }
            problem.y[i] = Double.parseDouble((String) inst.getTarget());
            problem.x[i] = x;
        }
        return problem;
    }

    /**
     * like toSparseMatrix but adds BOW as extra features
     *
     * @param instances
     * @return
     * @throws IOException
     */
    public svm_problem toSparseMatrixIncludeBOW(InstanceList instances) throws IOException {
        svm_problem problem = new svm_problem();
        problem.l = instances.size();
        problem.y = new double[problem.l];
        problem.x = new svm_node[problem.l][];
        for (int i = 0; i < instances.size(); i++) { //one instance is one document
            Instance inst = instances.get(i);
            FeatureVector tokens = (FeatureVector) inst.getData();
            int[] indices = tokens.getIndices();
            double[] values = tokens.getValues();
            HashMap<Integer, Integer> features = new HashMap<Integer, Integer>(); //map from cluster id (or feature id) to times of occurrences for an instance
            for (int k = 0; k < indices.length; k++) {
                if (wordIDsToFeatureIndices.containsKey(indices[k])) {
                    for (Integer cluster : wordIDsToFeatureIndices.get(indices[k])) {
                        if (!features.containsKey(cluster)) {
                            features.put(cluster, (int) values[k]);
                        } else {
                            features.put(cluster, features.get(cluster) + (int) values[k]);
                        }
                    }
                }
            }
            svm_node[] x = new svm_node[features.size() + indices.length]; // + bow size
            int k = 0;
            for (Map.Entry<Integer, Integer> entry : features.entrySet()) {
                x[k] = new svm_node();
                x[k].index = entry.getKey(); //cluster
                x[k].value = entry.getValue(); //number of occurrences of this cluster.
                k++;
            }
            //add bow features
            for (; k < indices.length + features.size(); k++) {
                x[k] = new svm_node();
                x[k].index = indices[k - features.size()];
                x[k].value = values[k - features.size()];
            }
            problem.y[i] = Double.parseDouble((String) inst.getTarget());
            problem.x[i] = x;
        }
        return problem;
    }


    public svm_problem toSparseMatrixTfidf(InstanceList instances) throws IOException {
        svm_problem problem = new svm_problem();
        problem.l = instances.size();
        problem.y = new double[problem.l];
        problem.x = new svm_node[problem.l][];

        HashMap<Integer, HashSet<Instance>> mapPathToDocuments = new HashMap<Integer, HashSet<Instance>>(); //map cluster path (index) -> hashset<instances>
        //todo save this a class member and if it is not null then it is the test set, use the same scales.

        for (int i = 0; i < instances.size(); i++) { //one instance is one document
            Instance inst = instances.get(i);
            FeatureVector tokens = (FeatureVector) inst.getData();
            int[] indices = tokens.getIndices();
            double[] values = tokens.getValues();
            HashMap<Integer, Integer> features = new HashMap<Integer, Integer>(); //One for every document. map from cluster to times of occurences for an instance
            for (int k = 0; k < indices.length; k++) { //for every word in the document
                if (wordIDsToFeatureIndices.containsKey(indices[k])) {
                    for (Integer cluster : wordIDsToFeatureIndices.get(indices[k])) {
                        if (!features.containsKey(cluster)) {
                            features.put(cluster, (int) values[k]);
                            if (!mapPathToDocuments.containsKey(cluster)) {
                                mapPathToDocuments.put(cluster, new HashSet<Instance>());
                            }
                            mapPathToDocuments.get(cluster).add(inst); // add document to the document freq map

                        } else {
                            features.put(cluster, features.get(cluster) + (int) values[k]);
                        }
                    }
                }
            }
        }
        for (int i = 0; i < instances.size(); i++) { //one instance is one document
            Instance inst = instances.get(i);
            FeatureVector tokens = (FeatureVector) inst.getData();
            int[] indices = tokens.getIndices();
            double[] values = tokens.getValues();
            HashMap<Integer, Integer> features = new HashMap<Integer, Integer>(); //One for every document. map from cluster to times of occurences for an instance
            for (int k = 0; k < indices.length; k++) {
                if (wordIDsToFeatureIndices.containsKey(indices[k])) {
                    for (Integer cluster : wordIDsToFeatureIndices.get(indices[k])) {
                        if (!features.containsKey(cluster)) {
                            features.put(cluster, (int) values[k]);
                        } else {
                            features.put(cluster, features.get(cluster) + (int) values[k]);
                        }
                    }
                }
            }
            svm_node[] x = new svm_node[features.size()];
            int k = 0;
            for (Map.Entry<Integer, Integer> entry : features.entrySet()) {
                double w_tf = 1 + Math.log10((double) entry.getValue());
                double N = (double) instances.size();
                double df = (double) mapPathToDocuments.get(entry.getKey()).size();
                double w_idf = Math.log10(N / df);
                double w = w_tf * w_idf;
                x[k] = new svm_node();
                x[k].index = entry.getKey(); //cluster
                x[k].value = w; //number of occurrences of this cluster.
                k++;
            }
            problem.y[i] = Double.parseDouble((String) inst.getTarget());
            problem.x[i] = x;
        }
        return problem;
    }


    public svm_problem toSparseMatrixIncludeBowTfidf(InstanceList instances) throws IOException {
        svm_problem problem = new svm_problem();
        problem.l = instances.size();
        problem.y = new double[problem.l];
        problem.x = new svm_node[problem.l][];

        HashMap<Integer, HashSet<Instance>> mapPathToDocuments = new HashMap<Integer, HashSet<Instance>>(); //map cluster path (index) -> hashset<instances>
        HashMap<Integer, HashSet<Instance>> wordIDToDocuments = new HashMap<Integer, HashSet<Instance>>();

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
                if (wordIDsToFeatureIndices.containsKey(indices[k])) {
                    for (Integer cluster : wordIDsToFeatureIndices.get(indices[k])) {
                        if (!features.containsKey(cluster)) {
                            features.put(cluster, (int) values[k]);
                            if (!mapPathToDocuments.containsKey(cluster)) {
                                mapPathToDocuments.put(cluster, new HashSet<Instance>());
                            }
                            mapPathToDocuments.get(cluster).add(inst); // add document to the document freq map
                        } else {
                            features.put(cluster, features.get(cluster) + (int) values[k]);
                        }
                    }
                }
            }
        }
        for (int i = 0; i < instances.size(); i++) { //one instance is one document
            Instance inst = instances.get(i);
            FeatureVector tokens = (FeatureVector) inst.getData();
            int[] indices = tokens.getIndices();
            double[] values = tokens.getValues();
            HashMap<Integer, Integer> features = new HashMap<Integer, Integer>(); //map from cluster id (or feature id) to times of occurrences for an instance
            for (int k = 0; k < indices.length; k++) {
                if (wordIDsToFeatureIndices.containsKey(indices[k])) {
                    for (Integer cluster : wordIDsToFeatureIndices.get(indices[k])) {
                        if (!features.containsKey(cluster)) {
                            features.put(cluster, (int) values[k]);
                        } else {
                            features.put(cluster, features.get(cluster) + (int) values[k]);
                        }
                    }
                }
            }
            svm_node[] x = new svm_node[features.size() + indices.length]; // + bow size
            int k = 0;
            for (Map.Entry<Integer, Integer> entry : features.entrySet()) {
                double w_tf = 1 + Math.log10((double) entry.getValue());
                double N = (double) instances.size();
                double df = (double) mapPathToDocuments.get(entry.getKey()).size();
                double w_idf = Math.log10(N / df);
                double w = w_tf * w_idf;
                x[k] = new svm_node();
                x[k].index = entry.getKey(); //cluster
                x[k].value = w; //number of occurrences of this cluster.
                k++;
            }
            //add bow features
            for (; k < indices.length + features.size(); k++) {
                double w_tf = 1 + Math.log10((double) values[k - features.size()] );
                double N = (double) instances.size();
                double df = (double) wordIDToDocuments.get(indices[k - features.size()]).size();
                double w_idf = Math.log10(N / df);
                double w = w_tf * w_idf;
                x[k] = new svm_node();
                x[k].index = indices[k - features.size()];
                x[k].value = w;
            }
            problem.y[i] = Double.parseDouble((String) inst.getTarget());
            problem.x[i] = x;
        }
        return problem;
    }


}
