package ma.emsi.hamidi.tp4hamidiweb.llm;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.data.message.SystemMessage;

import jakarta.enterprise.context.ApplicationScoped;


@ApplicationScoped
public class LlmClient {

    private String systemRole;
    private ChatMemory chatMemory;
    private Assistant assistant;

    /**
     * Constructeur : initialise le modèle Gemini et le service IA
     */
    public LlmClient() {

        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "Clé API manquante. Définissez GEMINI_API_KEY  dans vos variables d'environnement."
            );
        }

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .build();

    }

    /**
     * Définit le rôle système du LLM et réinitialise la mémoire.
     *
     * @param role rôle système choisi par l’utilisateur.
     */
    public void setSystemRole(String role) {
        this.systemRole = role;
        chatMemory.clear();
        if (role != null && !role.isBlank()) {
            chatMemory.add(SystemMessage.from(role));
        }
    }

    /**
     * Envoie un prompt (question) au LLM et renvoie la réponse.
     */
    public String ask(String prompt) {
        return assistant.chat(prompt);
    }

    public String getSystemRole() {
        return systemRole;
    }
}