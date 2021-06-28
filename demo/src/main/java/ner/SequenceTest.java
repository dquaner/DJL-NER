package ner;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import translators.SequenceTranslator;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;

public class SequenceTest {
    public static final String modelName = "dbmdz";
    public static void main(String[] args) throws IOException, MalformedModelException, ModelNotFoundException {
        String sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, " +
                "therefore very close to the Manhattan Bridge which is visible from the window.";

        SequenceTranslator translator = new SequenceTranslator(modelName);
        Criteria<String, String[][]> criteria = Criteria.builder()
                .setTypes(String.class, String[][].class)
                .optModelPath(Paths.get("models/" + modelName + "/"))
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();
        String[][] predictResult = null;
        try (Predictor<String, String[][]> predictor = ModelZoo.loadModel(criteria).newPredictor(translator)) {
            predictResult = predictor.predict(sequence);
        } catch (TranslateException e) {
            e.printStackTrace();
        }
        for (String[] tokens: predictResult) {
            System.out.println(tokens[0] + " " + tokens[1]);
        }
    }
}
