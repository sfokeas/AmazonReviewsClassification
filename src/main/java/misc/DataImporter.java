package misc;

import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.Alphabet;
import cc.mallet.types.InstanceList;

import java.io.*;
import java.util.ArrayList;
import java.util.regex.Pattern;

/**
 * The DataImporter class is important to parse data and create instances.
 * * Instances can be considered as tuples of (Name, Label, Data).
 * There is a one ot one relation between documents and instances. One instance corresponds to one document in the
 * data-set.
 */
public class DataImporter {
    Pipe pipe;
    Config config;

    public DataImporter(Config conf) {
        config = conf;
    }

    /**
     * Build the tokenization pipeline
     * @param alphabet
     * @return
     */
    private Pipe buildPipePreprocessed(Alphabet alphabet) {
        ArrayList pipeList = new ArrayList();

        pipeList.add(new Input2CharSequence("UTF-8"));

        //define the pattern for tokenization. The files have been preprocessed by the
        //preprocess scripts so it just splits on whitespace
        Pattern tokenPattern = Pattern.compile("[\\S+]+");
        pipeList.add(new CharSequence2TokenSequence(tokenPattern));

        if (config.getProperty("data.removeStopwords","false").trim().equalsIgnoreCase("true")) {
            // Remove stopwords from a standard English stoplist.
            //  options: [case sensitive] [mark deletions]
            pipeList.add(new TokenSequenceRemoveStopwords(false, false));
        }

        // Rather than storing tokens as strings, convert
        //  them to integers by looking them up in an alphabet.
        pipeList.add(new TokenSequence2FeatureSequence(alphabet));

        // Now convert the sequence of features to a sparse vector,
        //  mapping feature IDs to counts.
        pipeList.add(new FeatureSequence2FeatureVector());

        // Print out the features and the label. for debugging!
        //pipeList.add(new PrintInputAndTarget());

        return new SerialPipes(pipeList);
    }

    public InstanceList importPreprocessed(Alphabet alphabet, File inputFile) throws IOException {
        pipe = buildPipePreprocessed(alphabet);
        InstanceList instances = readFile(inputFile);
        return instances;
    }

    /**
     * Reads a data vile and produces instances
     * @param inputFile
     * @return
     * @throws IOException
     */
    private InstanceList readFile(File inputFile) throws IOException {
        //
        // Create the instance list and open the input file
        //
        String lineRegex = "^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$";
        Reader fileReader;
        fileReader = new InputStreamReader(new FileInputStream(inputFile));
        CsvIterator iterator = new CsvIterator(fileReader, Pattern.compile(lineRegex), 3, 2, 1);
        InstanceList instances = new InstanceList(pipe);
        instances.addThruPipe(iterator);
        System.out.println("finished creating instances");
        return instances;
    }


}
