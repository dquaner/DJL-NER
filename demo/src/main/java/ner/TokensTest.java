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
import java.util.Arrays;
import java.util.List;

public class TokensTest {
    public static void main(String[] args) throws Exception {
        Predictor<String[], String[]> predictor = getPredictor("elastic");
        List<String[][]> data = getTestData();
        System.out.println(data.size());
        List<String> predict = new ArrayList<>();
        List<String> label = new ArrayList<>();
        int count = 1;
        for (String[][] sequence: data) {
            System.out.println("------------ sequence(" +  count + ") ------------");
            label.addAll(Arrays.asList(sequence[1]));
            String[] result = predictor.predict(sequence[0]);
            predict.addAll(Arrays.asList(result));
            count ++;
        }
        System.out.println(predict.size());
        System.out.println(label.size());
        String[] entities = new String[]{"LOC", "ORG", "PER", "MISC"};
        double[] total_metrics = new double[]{0.0, 0.0, 0.0, 0.0, 0.0};
        for (String entity: entities) {
            double[] metrics = evaluate(entity, predict.toArray(new String[0]), label.toArray(new String[0]));
            for (int i=0; i<total_metrics.length; i++) {
                total_metrics[i] += metrics[i];
            }
            System.out.println(entity + ": " + metrics[3] + ", " + metrics[4] + ", " + metrics[5]);
        }
        total_metrics[3] /= entities.length;
        total_metrics[4] /= entities.length;
        double micro_precision = (total_metrics[0] + total_metrics[1]) == 0 ?
                Double.POSITIVE_INFINITY : total_metrics[0] / (total_metrics[0] + total_metrics[1]);
        double micro_recall = (total_metrics[0] + total_metrics[2]) == 0 ?
                Double.POSITIVE_INFINITY : total_metrics[0] / (total_metrics[0] + total_metrics[2]);
        double micro_f1 = (micro_precision == Double.POSITIVE_INFINITY || micro_recall == Double.POSITIVE_INFINITY) ?
                Double.POSITIVE_INFINITY : 2 * micro_precision * micro_recall / (micro_precision + micro_recall);
        System.out.println("micro_f1: " + micro_f1);

        System.out.println("macro_f1: " + 2 * total_metrics[3] * total_metrics[4] / (total_metrics[3] + total_metrics[4]));
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

    public static double[] evaluate(String entity, String[] predict, String[] label) {
        double[] metrics = new double[6];
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < predict.length; i++) {
            if (predict[i].endsWith(entity) && label[i].endsWith(entity)) {
                tp++;
            } else if (predict[i].endsWith(entity) && !label[i].endsWith(entity)) {
                fp++;
            } else if (!predict[i].endsWith(entity) && label[i].endsWith(entity)) {
                fn++;
            }
        }
        metrics[0] = tp;
        metrics[1] = fp;
        metrics[2] = fn;
        metrics[3] = (tp + fp) == 0 ? Double.POSITIVE_INFINITY : tp / (double) (tp + fp);
        metrics[4] = (tp + fn) == 0 ? Double.POSITIVE_INFINITY : tp / (double) (tp + fn);
        if (metrics[3] == Double.POSITIVE_INFINITY || metrics[4] == Double.POSITIVE_INFINITY) {
            metrics[5] = Double.POSITIVE_INFINITY;
        } else {
            metrics[5] = 2 * metrics[3] * metrics[4] / (metrics[3] + metrics[4]);
        }
        return metrics;
    }

}
