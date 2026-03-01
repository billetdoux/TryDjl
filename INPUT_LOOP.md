# Input Loop for Cosine Similarity

## Overview

The `Main.java` now includes an interactive loop that reads pairs of sentences from standard input and computes their cosine similarity using the embedding model.

## How It Works

### Loop Flow

```
1. Load the embedding model from local ONNX files
2. Create a custom translator to handle tokenization
3. Start interactive input loop:
   - Prompt for s1 (first sentence)
   - Prompt for s2 (second sentence)
   - Embed both sentences
   - Calculate cosine similarity
   - Print the result
   - Repeat until user types 'quit'
```

### Input Requirements

The loop waits for user input with these prompts:
```
s1: [user enters first sentence]
s2: [user enters second sentence]
Similarity: [computed result]
```

To exit, type `quit` at either prompt.

## Usage Example

```
Enter pairs of sentences (s1 and s2 on separate lines, or type 'quit' to exit):
s1: The quick brown fox jumps
s2: A fast red fox leaps
Similarity: 0.8234567

s1: Hello world
s2: Goodbye world
Similarity: 0.5123456

s1: quit
Done.
```

## Key Features

✅ **Continuous Input** - Reads pairs until user types 'quit'
✅ **Empty Line Handling** - Skips empty inputs and re-prompts
✅ **Error Handling** - Handles EOF (null input)
✅ **Resource Cleanup** - Closes predictor, model, and reader properly
✅ **Custom Translator** - Uses `MyTextEmbeddingTranslator` for ONNX compatibility

## Code Structure

### Main Loop
```java
while (true) {
    // Read s1
    System.out.print("s1: ");
    line = reader.readLine();
    if (line == null || line.equalsIgnoreCase("quit")) break;
    String s1 = line.trim();
    if (s1.isEmpty()) continue;
    
    // Read s2
    System.out.print("s2: ");
    line = reader.readLine();
    if (line == null || line.equalsIgnoreCase("quit")) break;
    String s2 = line.trim();
    if (s2.isEmpty()) continue;
    
    // Compute similarity
    float[] emb1 = predictor.predict(s1);
    float[] emb2 = predictor.predict(s2);
    double similarity = cosine(emb1, emb2);
    
    System.out.println("Similarity: " + similarity);
}
```

## Cosine Similarity Function

```java
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
```

Returns a value between -1.0 and 1.0:
- **1.0** = identical vectors
- **0.0** = orthogonal vectors
- **-1.0** = opposite vectors

## Running the Application

### In IntelliJ IDEA

1. Open `Main.java`
2. Click the Run button or press `Ctrl+Shift+F10`
3. Type sentences in the Console tab:
   ```
   s1: your first sentence
   s2: your second sentence
   ```
4. View the similarity score
5. Repeat or type `quit` to exit

### From Command Line

```bash
cd C:\Users\shah\IdeaProjects\TryDjl
mvn compile
mvn exec:java -Dexec.mainClass="org.examples.Test"
```

Then type your sentences as prompted.

## Model Information

- **Model**: all-MiniLM-L6-v2 (ONNX format)
- **Embedding Dimension**: 384
- **Pooling**: Mean pooling with L2 normalization
- **Engine**: OnnxRuntime
- **Tokenizer**: HuggingFaceTokenizer from local files

## Performance Notes

- **First sentence**: ~100-200ms (model warm-up)
- **Subsequent sentences**: ~50-100ms each
- **Similarity calculation**: <1ms
- **Model size**: ~50MB (loaded into memory once)

## Customization

### Change Pooling Strategy

Edit this line in `main()`:
```java
new MyTextEmbeddingTranslator(tokenizer, Batchifier.STACK, "mean", true, true);
                                                               ^^^^^^
```

Available options: `"mean"`, `"max"`, `"cls"`, `"weightedmean"`, `"mean_sqrt_len"`

### Disable L2 Normalization

Change:
```java
new MyTextEmbeddingTranslator(tokenizer, Batchifier.STACK, "mean", true, true);
                                                                     ^^^^
```
To: `false` (fourth parameter)

### Exclude Token Type IDs

Change:
```java
new MyTextEmbeddingTranslator(tokenizer, Batchifier.STACK, "mean", true, true);
                                                                          ^^^^
```
To: `false` (fifth parameter)

## Troubleshooting

### "Input mismatch" Error
Ensure the custom `MyTextEmbeddingTranslator` is properly tokenizing and providing `input_ids`, `attention_mask`, and `token_type_ids`.

### Slow Embeddings
First embedding is slow due to model loading. Subsequent ones are faster.

### Memory Issues
The model uses ~500MB-1GB. Ensure sufficient heap space.

## Summary

The input loop makes it easy to interactively compare sentence similarities without modifying code. Perfect for testing and exploration!

