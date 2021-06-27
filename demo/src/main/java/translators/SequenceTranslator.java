package translators;

import ai.djl.Model;
import ai.djl.modality.nlp.SimpleVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertTokenizer;
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
import java.util.List;

public class SequenceTranslator implements Translator<String, String[][]> {
    private List<String> tokens;
    private final String modelName;
    private Vocabulary vocabulary;
    private BertTokenizer tokenizer;

    public SequenceTranslator(String modelName) {
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
        tokenizer = new BertTokenizer();
    }

    @Override
    public String[][] processOutput(TranslatorContext translatorContext, NDList ndList) throws Exception {
        ObjectMapper objectMapper = new ObjectMapper();
        JsonNode id2label = objectMapper.readTree(new File("models/" + modelName + "/config.json")).get("id2label");
        String[][] result = new String[tokens.size()][2];
        Number[] indices = ndList.get(0).argMax(1).toArray();
        for (int i = 0; i < tokens.size(); i++) {
            result[i] = new String[]{tokens.get(i), id2label.get(String.valueOf(indices[i])).asText()};
        }
        return result;
    }

    @Override
    public NDList processInput(TranslatorContext translatorContext, String sequence) {
        // get the encoded tokens that would be used in precessOutput
        tokens = tokenizer.tokenize(sequence);
        NDManager manager = translatorContext.getNDManager();
        // map the tokens(String) to indices(long)
        long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();
        NDArray indicesArray = manager.create(indices);
        // The order matters
        return new NDList(indicesArray);
    }
}
