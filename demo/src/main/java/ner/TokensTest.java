package ner;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.training.util.ProgressBar;
import translators.TokensTranslator;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class TokensTest {
    public static void main(String[] args) throws Exception {
        Predictor<String[], String[]> predictor = getPredictor("tiny-dbmdz");
        List<String[][]> data = getTestData();
        System.out.println(data.size());
        int count = 1;
        for (String[][] sequence: data) {
            if (count > 10) {
                break;
            }
            System.out.println("------------ sequence(" +  count + ") ------------");
            String[] result = predictor.predict(sequence[0]);
            for (int i = 0; i < result.length; i++) {
                System.out.println(sequence[1][i] + " -> " + result[i]);
            }
            count ++;
        }


    }

    public static List<String[][]> getTestData() throws IOException {
        List<String> lines = Files.readAllLines(Paths.get("data/conll2003test"));
        List<String[][]> data = new ArrayList<>();
        String[][] sequence = new String[2][];
        List<String> tokens = new ArrayList<>();
        List<String> entities = new ArrayList<>();
        for(String line: lines) {
            if (line.length() == 0 && tokens.size() > 256) {
                sequence[0] = tokens.toArray(new String[0]);
                sequence[1] = entities.toArray(new String[0]);
                data.add(sequence);
                sequence = new String[2][];
                tokens = new ArrayList<>();
                entities = new ArrayList<>();
            } else if (line.length() > 0) {
                tokens.add(line.split(" ")[0]);
                entities.add(line.split(" ")[3]);
            }
        }
        return data;
    }

    public static Predictor<String[], String[]> getPredictor(String modelName) throws Exception {
        TokensTranslator translator = new TokensTranslator(modelName);
        Criteria<String[], String[]> criteria = Criteria.builder()
                .setTypes(String[].class, String[].class)
                .optModelPath(Paths.get("models/" + modelName + "/"))
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();
        return ModelZoo.loadModel(criteria).newPredictor(translator);
    }
}
