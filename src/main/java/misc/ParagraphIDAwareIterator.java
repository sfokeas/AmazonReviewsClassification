package misc;

import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by sotos on 4/30/16.
 */
public class ParagraphIDAwareIterator implements LabelAwareIterator {
    protected LabelsSource labelsSource;
    String parPrefix = "paragraphID_";
    int position;
    InstanceList instances;

    public ParagraphIDAwareIterator(InstanceList InputInstances) {
        instances = InputInstances;
        labelsSource = getAllParagraphIDs();
        position = 0;
    }

    private LabelsSource getAllParagraphIDs() {
        List<String> labels = new ArrayList();
        for (int i = 0; i < instances.size(); i++) { //each instance corresponds to a document
            Instance inst = instances.get(i);
            String paragraphID = (String) inst.getName();
            paragraphID = parPrefix + paragraphID;
            labels.add(paragraphID);
        }
        return new LabelsSource(labels);
    }

    @Override
    public LabelsSource getLabelsSource() {
        return labelsSource;
    }

    @Override
    public void reset() {
        position = 0;
    }

    public boolean hasNextDocument() {
        int size = instances.size();
        return position < size;
    }

    public LabelledDocument nextDocument() {
        Instance inst = instances.get(position);
        position++;
        LabelledDocument document = new LabelledDocument();
        String paragraphID = (String) inst.getName();
        paragraphID = parPrefix + paragraphID;
        document.setLabel(paragraphID);
        String content = Arrays.toString(((FeatureVector) inst.getData()).getIndices())
                .replace('[', ' ')
                .replace(']', ' ')
                .replace(',', ' ').trim();
        document.setContent(content);
        return document;
    }

}