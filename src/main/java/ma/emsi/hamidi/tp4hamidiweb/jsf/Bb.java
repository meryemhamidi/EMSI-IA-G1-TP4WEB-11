package ma.emsi.hamidi.tp4hamidiweb.jsf;

import jakarta.faces.application.FacesMessage;
import jakarta.faces.context.FacesContext;
import jakarta.faces.model.SelectItem;
import jakarta.faces.view.ViewScoped;
import jakarta.inject.Inject;
import jakarta.inject.Named;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import ma.emsi.hamidi.tp4hamidiweb.llm.LlmClient;

/**
 * Backing bean pour la page JSF index.xhtml.
 * Portée view pour conserver l'état de la conversation (chat).
 */
@Named("bb")
@ViewScoped
public class Bb implements Serializable {

    private String roleSysteme;
    private boolean roleSystemeChangeable = true;

    private String question;
    private String reponse;
    private StringBuilder conversation = new StringBuilder();

    @Inject
    private LlmClient llm;

    /** Bouton "Envoyer" */
    public String envoyer() {
        if (question == null || question.isBlank()) {
            addMsg(FacesMessage.SEVERITY_ERROR, "Texte question vide", "Il manque le texte de la question.");
            return null;
        }

        try {
            if (roleSystemeChangeable) {
                llm.setSystemRole(roleSysteme);
                roleSystemeChangeable = false;
            }

            reponse = llm.ask(question);

            // Historique affiché dans la zone "conversation"
            appendConversation(question, reponse);

        } catch (Exception e) {
            reponse = null;
            addMsg(FacesMessage.SEVERITY_ERROR, "Erreur LLM", e.getMessage());
        }
        return null; // rester sur la même vue
    }

    /** Bouton "Nouveau chat" : on repart de zéro (nouvelle instance @ViewScoped) */
    public String nouveauChat() {
        // Redirection pour forcer une nouvelle vue → nouveau bean
        return "index?faces-redirect=true";
    }

    // --- Utilitaires ---
    private void appendConversation(String q, String r) {
        conversation
                .append("== User:\n").append(q).append("\n")
                .append("== Assistant:\n").append(r).append("\n\n");
    }

    private void addMsg(FacesMessage.Severity sev, String summary, String detail) {
        FacesContext.getCurrentInstance().addMessage(null, new FacesMessage(sev, summary, detail));
    }

    // --- Rôles prédéfinis pour la liste déroulante (optionnel si tu utilises un textarea libre) ---
    private List<SelectItem> listeRolesSysteme;
    public List<SelectItem> getRolesSysteme() {
        if (this.listeRolesSysteme == null) {
            // Génère les rôles de l'API prédéfinis
            this.listeRolesSysteme = new ArrayList<>();
            // Vous pouvez évidemment écrire ces rôles dans la langue que vous voulez.
            String role = """
                    You are a helpful assistant. You help the user to find the information they need.
                    If the user type a question, you answer it.
                    """;
            // 1er argument : la valeur du rôle, 2ème argument : le libellé du rôle
            this.listeRolesSysteme.add(new SelectItem(role, "Assistant"));

            role = """
                    You are an interpreter. You translate from English to French and from French to English.
                    If the user type a French text, you translate it into English.
                    If the user type an English text, you translate it into French.
                    If the text contains only one to three words, give some examples of usage of these words in English.
                    """;
            this.listeRolesSysteme.add(new SelectItem(role, "Traducteur Anglais-Français"));

            role = """
                    Your are a travel guide. If the user type the name of a country or of a town,
                    you tell them what are the main places to visit in the country or the town
                    are you tell them the average price of a meal.
                    """;
            this.listeRolesSysteme.add(new SelectItem(role, "Guide touristique"));

            role = """
                    You are a cooking assistant and nutritionist. If the user provides ingredients,
                    you suggest a recipe. If the user provides a recipe,
                    you calculate the nutritional information (calories, protein, etc.) for a standard serving.
                    You can also suggest healthy substitutions for ingredients.
                    """;
            this.listeRolesSysteme.add(new SelectItem(role, "Assistant en cuisine"));

        }

        return this.listeRolesSysteme;
    }

    // --- Getters/Setters pour la page JSF ---
    public String getRoleSysteme() { return roleSysteme; }
    public void setRoleSysteme(String roleSysteme) { this.roleSysteme = roleSysteme; }

    public boolean isRoleSystemeChangeable() { return roleSystemeChangeable; }

    public String getQuestion() { return question; }
    public void setQuestion(String question) { this.question = question; }

    public String getReponse() { return reponse; }
    public void setReponse(String reponse) { this.reponse = reponse; }

    public String getConversation() { return conversation.toString(); }
    public void setConversation(String conversation) { this.conversation = new StringBuilder(conversation); }
}