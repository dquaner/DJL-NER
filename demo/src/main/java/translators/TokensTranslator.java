package translators;

import ai.djl.Model;
import ai.djl.modality.nlp.SimpleVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class TokensTranslator implements Translator<String[], String[]> {
    private final String modelName;
    private Vocabulary vocabulary;

    public TokensTranslator(String modelName) {
        this.modelName = modelName;
    }

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }

    @Override
    public void prepare(NDManager manager, Model model) throws IOException {
        Path path = Paths.get("models/" + modelName + "/vocab.txt");
        vocabulary = SimpleVocabulary.builder()
                .optMinFrequency(1)
                .addFromTextFile(path)
                .optUnknownToken("[UNK]")
                .build();
    }

    @Override
    public String[] processOutput(TranslatorContext translatorContext, NDList ndList) throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();
        JsonNode id2label = objectMapper.readTree(new File("models/" + modelName + "/config.json")).get("id2label");
        Number[] indices = ndList.get(0).argMax(1).toArray();
        String[] result = new String[indices.length];
        for( int i = 0; i < indices.length; i++) {
            result[i] = id2label.get(String.valueOf(indices[i])).asText();
        }
        return result;
    }

    @Override
    public NDList processInput(TranslatorContext translatorContext, String[] tokens) {
        NDManager manager = translatorContext.getNDManager();
        // map the tokens(String) to indices(long)
        long[] indices = Arrays.stream(tokens).mapToLong(vocabulary::getIndex).toArray();
        NDArray indicesArray = manager.create(indices);
        return new NDList(indicesArray);
    }
}
