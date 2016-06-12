package misc;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;

import java.util.regex.Pattern;

/**
 * Created by sotos on 5/1/16.
 */
public class DummyPreprocessor implements TokenPreProcess {

    @Override
    public String preProcess(String token) {
        return token;
    }
}
