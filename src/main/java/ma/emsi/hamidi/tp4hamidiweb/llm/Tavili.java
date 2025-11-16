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
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import java.net.URL;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Tavili {

    private RetrievalAugmentor augmentor;

    @PostConstruct
    void init() {
        try {
            var parser = new ApacheTikaDocumentParser();
            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

            ContentRetriever retrRag = buildRetriever("documents/RAG.pdf", parser, embeddingModel);
            ContentRetriever retrMonumentsEurope = buildRetriever("documents/MonumentsEurope.pdf", parser, embeddingModel);

            String tavilyKey = System.getenv("TAVILY_API_KEY");
            if (tavilyKey == null || tavilyKey.isBlank()) {
                throw new IllegalStateException("TAVILY_API_KEY manquante !");
            }

            TavilyWebSearchEngine tavilyEngine = TavilyWebSearchEngine.builder()
                    .apiKey(tavilyKey)
                    .build();

            ContentRetriever retrWeb = WebSearchContentRetriever.builder()
                    .webSearchEngine(tavilyEngine)
                    .maxResults(5)
                    .build();

            String geminiKey = System.getenv("GEMINI_API_KEY");
            ChatModel model = GoogleAiGeminiChatModel.builder()
                    .apiKey(geminiKey)
                    .temperature(0.3)
                    .logRequestsAndResponses(true)
                    .modelName("gemini-2.5-flash")
                    .build();

            Map<ContentRetriever, String> sources = new HashMap<>();
            sources.put(retrRag, "Documents sur le RAG, le fine-tuning et l’IA");
            sources.put(retrMonumentsEurope, "Documents décrivant des monuments, architecture, patrimoine culturel et historique");
            sources.put(retrWeb, "Recherche web via Tavily");

            QueryRouter router = new DefaultQueryRouter(List.of(retrRag, retrMonumentsEurope, retrWeb));
            augmentor = DefaultRetrievalAugmentor.builder()
                    .queryRouter(router)
                    .build();

        } catch (Exception e) {
            throw new RuntimeException("Erreur d’initialisation du RAG avec Tavily", e);
        }
    }

    public RetrievalAugmentor getAugmentor() {
        return augmentor;
    }

    private ContentRetriever buildRetriever(String name, ApacheTikaDocumentParser parser, EmbeddingModel model) {
        try {
            URL res = getClass().getClassLoader().getResource(name);
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
            throw new RuntimeException(e);
        }
    }
}