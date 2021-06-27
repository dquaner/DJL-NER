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
import java.util.ArrayList;
import java.util.List;

public class TokensTranslator implements Translator<List<String>, List<String[]>> {
    private List<String> tokens;
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
    public List<String[]> processOutput(TranslatorContext translatorContext, NDList ndList) throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();
        JsonNode id2label = objectMapper.readTree(new File("models/" + modelName + "/config.json")).get("id2label");
        NDArray logistics = ndList.get(0);
        List<String[]> result = new ArrayList<>();
        Number[] indices = logistics.argMax(1).toArray();
        for (int i = 0; i < tokens.size(); i++) {
            result.add(new String[]{tokens.get(i), id2label.get(String.valueOf(indices[i])).asText()});
        }
        return result;
    }

    @Override
    public NDList processInput(TranslatorContext translatorContext, List<String> tokens) {
        // get the encoded tokens that would be used in precessOutput
        this.tokens = tokens;
        NDManager manager = translatorContext.getNDManager();
        // map the tokens(String) to indices(long)
        long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();
        NDArray indicesArray = manager.create(indices);
        // The order matters
        return new NDList(indicesArray);
    }
}
