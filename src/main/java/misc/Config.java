package misc;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;

/**
 * Created by sotos on 3/9/16.
 */
public class Config extends Properties {
    //add functionality for configuration sanity checks

    //the constructor can check if the file is properly formatted and throw an exception. built your own exception?
    public Config(String configFile) throws IOException {

        //load?
        this.load(new InputStreamReader(new FileInputStream(configFile)));
        checkFormat();
    }

    private boolean checkFormat() {
        // if config.get property is not set ...
        // throw new ConfigFileException();
        //TODO
        return true;
    }

    public static String returnOutputNamePrefix(Config config) {  //todo use this everywhere instead of hardcoding
        double C = Double.parseDouble(config.getProperty("clf.svm.C", "1"));
        String namePrefix = null;
        Path pData = Paths.get(config.getProperty("data.input"));
        String inputFilename = pData.getFileName().toString();
        if (config.getProperty("data.model").trim().equals("brown")) {
            Path pBrown = Paths.get(config.getProperty("data.brown.paths"));
            String brownPathsName = pBrown.getName(pBrown.getNameCount() - 2).toString();
            namePrefix = config.getProperty("general.outputDir") + "/"
                    + inputFilename
                    + "_" + config.getProperty("data.model")
                    + "_" + brownPathsName
                    + "_C_" + C;
        } else if (config.getProperty("data.model").trim().equals("word2vec")) {
            if (config.getProperty("w2v.lookupTable", "").trim().equalsIgnoreCase("")) {
                double learningRate = Double.parseDouble(config.getProperty("w2v.learningRate", "0.025"));
                //double minLearningRate = Double.parseDouble(config.getProperty("w2v.MinLearningRate", "0.000001"));
                int batchSize = Integer.parseInt(config.getProperty("w2v.batchSize", "1000"));
                int epochs = Integer.parseInt(config.getProperty("w2v.epochs", "1000"));
                boolean trainWordVectors = Boolean.parseBoolean(config.getProperty("w2v.trainWordVectors", "false"));
                int layerSize = Integer.parseInt(config.getProperty("w2v.layerSize", "100"));
                int windowSize = Integer.parseInt(config.getProperty("w2v.windowSize", "5"));
                int minWordFrequency = Integer.parseInt(config.getProperty("w2v.minWordFrequency", "1"));
                double subSampling = Double.parseDouble(config.getProperty("w2v.subSampling", "0"));
                double negSampling = Double.parseDouble(config.getProperty("w2v.negSampling", "0"));
                namePrefix = config.getProperty("general.outputDir") + "/"
                        + inputFilename
                        + "_" + config.getProperty("data.model")
                        + "_C_" + C
                        + "_learningRate_" + learningRate
                        + "_batchSize_" + batchSize
                        + "_epochs_" + epochs
                        + "_wordVectors_" + trainWordVectors
                        + "_layerSize_" + layerSize
                        + "_windowSize_" + windowSize
                        + "_wordFrequency_" + minWordFrequency
                        + "_subSampling_" + subSampling
                        + "_negSampling_" + negSampling;
            } else {
                Path pLookUpTable = Paths.get(config.getProperty("w2v.lookupTable"));
                String lookUpTableFileName = pLookUpTable.getName(pLookUpTable.getNameCount() - 2).toString();
                namePrefix = config.getProperty("general.outputDir") + "/"
                        + inputFilename
                        + "_" + config.getProperty("data.model")
                        + "_" + lookUpTableFileName
                        + "_C_" + C;
            }
        } else {
            namePrefix = config.getProperty("general.outputDir") + "/"
                    + inputFilename
                    + "_" + config.getProperty("data.model")
                    + "_C_" + C;
        }
        return namePrefix;
    }

}
