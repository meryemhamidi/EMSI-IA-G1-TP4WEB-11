package ma.emsi.hamidi.tp4hamidiweb.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;

import java.net.URL;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Routage {

    private RetrievalAugmentor augmentor;

    @PostConstruct
    void init() {
        var parser = new ApacheTikaDocumentParser();
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        ContentRetriever retrRag = buildRetriever("documents/RAG.pdf", parser, embeddingModel);
        ContentRetriever retrMonumentsEurope = buildRetriever("documents/MonumentsEurope.pdf", parser, embeddingModel);

        String cle = System.getenv("GEMINI_API_KEY");
        if (cle == null || cle.isEmpty()) {
            throw new IllegalStateException("GEMINI_API_KEY environment variable is not set");
        }

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(cle)
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .modelName("gemini-2.5-flash")
                .build();

        Map<ContentRetriever, String> sources = new HashMap<>();
        sources.put(retrRag, "Documents sur le RAG, le fine-tuning et l’intelligence artificielle");
        sources.put(retrMonumentsEurope, "Documents décrivant des monuments, architecture, patrimoine culturel et historique");

        QueryRouter router = new LanguageModelQueryRouter(model, sources);
        augmentor = DefaultRetrievalAugmentor.builder().queryRouter(router).build();
    }

    public RetrievalAugmentor getAugmentor() {
        return augmentor;
    }

    private ContentRetriever buildRetriever(String name, ApacheTikaDocumentParser parser, EmbeddingModel model) {
        try {
            URL res = getClass().getClassLoader().getResource(name);
            if (res == null) {
                throw new RuntimeException("Resource not found: " + name);
            }
            var path = Paths.get(res.toURI());
            Document doc = FileSystemDocumentLoader.loadDocument(path, parser);
            var segments = DocumentSplitters.recursive(300, 30).split(doc);
            List<Embedding> embs = model.embedAll(segments).content();
            EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
            store.addAll(embs, segments);
            return EmbeddingStoreContentRetriever.builder()
                    .embeddingStore(store)
                    .embeddingModel(model)
                    .maxResults(2)
                    .minScore(0.5)
                    .build();
        } catch (Exception e) {
            throw new RuntimeException("Error building retriever for " + name, e);
        }
    }
}
