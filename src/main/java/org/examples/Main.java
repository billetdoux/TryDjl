package org.examples;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;

class Test {

    static void main() throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
        Path path = Paths.get("C:\\Users\\shah\\IdeaProjects\\TryDjl\\src\\main\\resources\\models\\minilm");

        HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.builder()
                        .optTokenizerPath(path)
                        .optManager(NDManager.newBaseManager("PyTorch"))
                        .build();
        MyTextEmbeddingTranslator translator =
                new MyTextEmbeddingTranslator(tokenizer, Batchifier.STACK, "mean", true, true);

        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optApplication(Application.NLP.TEXT_EMBEDDING)
                .optModelPath(path)
                .optTranslator(translator)
                .optEngine("OnnxRuntime")
                .build();

        ZooModel<String, float[]> model = criteria.loadModel();
        Predictor<String, float[]> predictor = model.newPredictor();

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        String line;

        System.out.println("Enter pairs of sentences (s1 and s2 on separate lines, or type 'quit' to exit):");
        while (true) {
            System.out.print("s1: ");
            line = reader.readLine();
            if (line == null || line.equalsIgnoreCase("quit")) {
                break;
            }
            String s1 = line.trim();
            if (s1.isEmpty()) {
                continue;
            }

            System.out.print("s2: ");
            line = reader.readLine();
            if (line == null || line.equalsIgnoreCase("quit")) {
                break;
            }
            String s2 = line.trim();
            if (s2.isEmpty()) {
                continue;
            }

            float[] emb1 = predictor.predict(s1);
            float[] emb2 = predictor.predict(s2);
            double similarity = cosine(emb1, emb2);

            System.out.println("Similarity: " + similarity);
            System.out.println();
        }

        predictor.close();
        model.close();
        reader.close();
        System.out.println("Done.");
    }

    public static float cosine(float[] a, float[] b) {
        if (a.length != b.length)
            throw new IllegalArgumentException("Vector lengths differ");

        float dot = 0.0F;
        float normA = 0.0F;
        float normB = 0.0F;

        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        return (float) (dot / (Math.sqrt(normA) * Math.sqrt(normB)));
    }

    static final class MyTextEmbeddingTranslator implements Translator<String, float[]> {

        private static final int[] AXIS = {0};

        private HuggingFaceTokenizer tokenizer;
        private Batchifier batchifier;
        private boolean normalize;
        private String pooling;
        private boolean includeTokenTypes;

        MyTextEmbeddingTranslator(
                HuggingFaceTokenizer tokenizer,
                Batchifier batchifier,
                String pooling,
                boolean normalize,
                boolean includeTokenTypes) {
            this.tokenizer = tokenizer;
            this.batchifier = batchifier;
            this.pooling = pooling;
            this.normalize = normalize;
            this.includeTokenTypes = includeTokenTypes;
        }

        /** {@inheritDoc} */
        @Override
        public Batchifier getBatchifier() {
            return batchifier;
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            Encoding encoding = tokenizer.encode(input);
            ctx.setAttachment("encoding", encoding);
            return encoding.toNDList(ctx.getNDManager(), includeTokenTypes, false); // TODO false?
        }

        /** {@inheritDoc} */
        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            Encoding encoding = (Encoding) ctx.getAttachment("encoding");
            NDManager manager = ctx.getNDManager();
            NDArray embeddings = processEmbedding(manager, list, encoding, pooling);
            if (normalize) {
                embeddings = embeddings.normalize(2, 0);
            }

            return embeddings.toFloatArray();
        }

        static NDArray processEmbedding(
                NDManager manager, NDList list, Encoding encoding, String pooling) {
            NDArray embedding = list.get("last_hidden_state");
            if (embedding == null) {
                // For Onnx model, NDArray name is not present
                embedding = list.head();
            }
            long[] attentionMask = encoding.getAttentionMask();
            try (NDManager ptManager = NDManager.newBaseManager("PyTorch")) {
                NDArray inputAttentionMask = ptManager.create(attentionMask).toType(DataType.FLOAT32, true);
                switch (pooling) {
                    case "mean":
                        return meanPool(embedding, inputAttentionMask, false);
                    case "mean_sqrt_len":
                        return meanPool(embedding, inputAttentionMask, true);
                    case "max":
                        return maxPool(embedding, inputAttentionMask);
                    case "weightedmean":
                        return weightedMeanPool(embedding, inputAttentionMask);
                    case "cls":
                        return embedding.get(0);
                    default:
                        throw new AssertionError("Unexpected pooling mode: " + pooling);
                }
            }
        }

        private static NDArray meanPool(NDArray embeddings, NDArray attentionMask, boolean sqrt) {
            long[] shape = embeddings.getShape().getShape();
            attentionMask = attentionMask.expandDims(-1).broadcast(shape);
            NDArray inputAttentionMaskSum = attentionMask.sum(AXIS);
            NDArray clamp = inputAttentionMaskSum.clip(1e-9, 1e12);
            NDArray prod = embeddings.mul(attentionMask);
            NDArray sum = prod.sum(AXIS);
            if (sqrt) {
                return sum.div(clamp.sqrt());
            }
            return sum.div(clamp);
        }

        private static NDArray maxPool(NDArray embeddings, NDArray inputAttentionMask) {
            long[] shape = embeddings.getShape().getShape();
            inputAttentionMask = inputAttentionMask.expandDims(-1).broadcast(shape);
            inputAttentionMask = inputAttentionMask.eq(0);
            embeddings = embeddings.duplicate();
            embeddings.set(inputAttentionMask, -1e9); // Set padding tokens to large negative value

            return embeddings.max(AXIS, true);
        }

        private static NDArray weightedMeanPool(NDArray embeddings, NDArray attentionMask) {
            long[] shape = embeddings.getShape().getShape();
            NDArray weight = embeddings.getManager().arange(1, shape[0] + 1);
            weight = weight.expandDims(-1).broadcast(shape);

            attentionMask = attentionMask.expandDims(-1).broadcast(shape).mul(weight);
            NDArray maskSum = attentionMask.sum(AXIS);
            NDArray embeddingSum = embeddings.mul(attentionMask).sum(AXIS);
            return embeddingSum.div(maskSum);
        }
    }

}