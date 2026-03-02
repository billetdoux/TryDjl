package org.examples;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.embedding.TextEmbedding;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.io.IOException;

public class BgeEmbeddingExample {

    static void main(String[] args) throws IOException, ModelException, TranslateException {

        String sentence1 = "The company will not issue dividends this year.";
        String sentence2 = "The company will issue dividends this year.";

        Criteria<String, float[]> criteria =
                Criteria.builder()
                        .setTypes(String.class, float[].class)
                        .optApplication(Application.NLP.TEXT_EMBEDDING)
                        .optModelUrls("djl://ai.djl.huggingface.pytorch/BAAI/bge-small-en-v1.5")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<String, float[]> model = ModelZoo.loadModel(criteria);
             Predictor<String, float[]> predictor = model.newPredictor()) {

            float[] embedding1 = predictor.predict(sentence1);
            float[] embedding2 = predictor.predict(sentence2);

            double cosineSimilarity = cosineSimilarity(embedding1, embedding2);

            System.out.println("Cosine similarity: " + cosineSimilarity);
        }
    }

    private static double cosineSimilarity(float[] v1, float[] v2) {
        double dot = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (int i = 0; i < v1.length; i++) {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
}
