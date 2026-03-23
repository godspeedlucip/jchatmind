package com.kama.jchatmind.agent.graph;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;

import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
public class SupervisorNode implements AgentNode {

    private static final Pattern ROUTE_PATTERN = Pattern.compile("(?im)^\\s*ROUTE\\s*:\\s*(WORKER|FINISH)\\s*$");
    private static final Pattern ANSWER_PATTERN = Pattern.compile("(?is)ANSWER\\s*:\\s*(.+)$");

    private final ChatClient chatClient;
    private final ChatOptions chatOptions;
    private final Set<String> availableToolNames;

    public SupervisorNode(ChatClient chatClient, ChatOptions chatOptions, Set<String> availableToolNames) {
        this.chatClient = chatClient;
        this.chatOptions = chatOptions;
        this.availableToolNames = availableToolNames;
    }

    @Override
    public String getName() {
        return "SUPERVISOR";
    }

    @Override
    public AgentGraphState process(AgentGraphState state) {
        log.info("[GraphEngine] Node: {}", getName());

        List<Message> messages = state.getMessages();
        int stepCount = getIntAttr(state, "step_count", 1);
        int workerHops = getIntAttr(state, "worker_hops", 0);
        String workerCandidateAnswer = getStringAttr(state, "worker_candidate_answer");

        boolean weatherIntent = isWeatherIntent(messages);
        boolean hasWeatherResult = hasToolResponseNamedInCurrentTurn(state, "weather");
        boolean latestIsToolResponse = hasLatestToolResponseInCurrentTurn(state);
        boolean hasToolFailure = hasRecentToolFailureInCurrentTurn(state);
        boolean hasKnowledgeTool = hasToolName("KnowledgeTool");

        String toolNames = availableToolNames == null || availableToolNames.isEmpty()
                ? "(none)"
                : String.join(", ", availableToolNames);

        String supervisorPrompt = """
                You are a supervisor in a multi-agent graph.

                Available tools: %s

                Output format (STRICT):
                ROUTE: WORKER or ROUTE: FINISH
                ANSWER: <only when ROUTE is FINISH>

                Rules:
                - Choose ROUTE: WORKER if additional tool calls are still required.
                - Choose ROUTE: FINISH only when the task is complete or cannot continue safely.
                - Final answer must be user-facing result only.
                - Never output DSML/XML/function-call markup.
                - Never invent tool names.
                """.formatted(toolNames);

        Prompt prompt = Prompt.builder()
                .chatOptions(chatOptions)
                .messages(messages)
                .build();

        ChatResponse response = chatClient.prompt(prompt)
                .system(supervisorPrompt)
                .call()
                .chatClientResponse()
                .chatResponse();

        String modelText = response.getResult().getOutput().getText();
        String text = modelText == null ? "" : modelText.trim();
        RouteDecision decision = parseDecision(text);

        if (stepCount <= 1 && hasKnowledgeTool && !hasToolFailure && !latestIsToolResponse) {
            state.getAttributes().put("worker_hops", workerHops + 1);
            state.setNextNode("WORKER");
            return state;
        }

        if (stepCount <= 1 && shouldForceWorkerByUserIntent(messages) && !hasToolFailure) {
            state.getAttributes().put("worker_hops", workerHops + 1);
            state.setNextNode("WORKER");
            return state;
        }

        if (weatherIntent && !hasWeatherResult && workerHops < 6) {
            state.getAttributes().put("worker_hops", workerHops + 1);
            state.setNextNode("WORKER");
            return state;
        }

        if (decision.route == Route.WORKER) {
            if (workerHops >= 6) {
                AssistantMessage finalMsg = new AssistantMessage(fallbackFinalAnswer(state));
                state.getMessages().add(finalMsg);
                state.getAttributes().put("latest_message", finalMsg);
                state.getAttributes().put("worker_hops", 0);
                state.setNextNode("FINISH");
                return state;
            }
            state.getAttributes().put("worker_hops", workerHops + 1);
            state.setNextNode("WORKER");
            return state;
        }

        if (decision.route == Route.FINISH) {
            String finalAnswer;
            if (latestIsToolResponse) {
                finalAnswer = synthesizeNaturalLanguageAnswer(state);
            } else {
                finalAnswer = decision.answer;
                if (finalAnswer.isBlank()) {
                    finalAnswer = !workerCandidateAnswer.isBlank()
                            ? workerCandidateAnswer
                            : synthesizeNaturalLanguageAnswer(state);
                }
            }
            if (finalAnswer.isBlank()) {
                finalAnswer = fallbackFinalAnswer(state);
            }
            AssistantMessage finalMsg = new AssistantMessage(stripProcessText(finalAnswer));
            state.getMessages().add(finalMsg);
            state.getAttributes().put("latest_message", finalMsg);
            state.getAttributes().put("worker_hops", 0);
            state.setNextNode("FINISH");
            return state;
        }

        if (hasToolFailure) {
            AssistantMessage finalMsg = new AssistantMessage(fallbackFinalAnswer(state));
            state.getMessages().add(finalMsg);
            state.getAttributes().put("latest_message", finalMsg);
            state.getAttributes().put("worker_hops", 0);
            state.setNextNode("FINISH");
            return state;
        }

        if (!workerCandidateAnswer.isBlank() && !latestIsToolResponse) {
            AssistantMessage finalMsg = new AssistantMessage(stripProcessText(workerCandidateAnswer));
            state.getMessages().add(finalMsg);
            state.getAttributes().put("latest_message", finalMsg);
            state.getAttributes().put("worker_hops", 0);
            state.setNextNode("FINISH");
            return state;
        }

        if (latestIsToolResponse) {
            String finalAnswer = synthesizeNaturalLanguageAnswer(state);
            if (finalAnswer.isBlank()) {
                finalAnswer = fallbackFinalAnswer(state);
            }
            AssistantMessage finalMsg = new AssistantMessage(stripProcessText(finalAnswer));
            state.getMessages().add(finalMsg);
            state.getAttributes().put("latest_message", finalMsg);
            state.getAttributes().put("worker_hops", 0);
            state.setNextNode("FINISH");
            return state;
        }

        state.getAttributes().put("worker_hops", workerHops + 1);
        state.setNextNode("WORKER");
        return state;
    }

    private String synthesizeNaturalLanguageAnswer(AgentGraphState state) {
        try {
            String latestUserQuestion = latestUserQuestionText(state.getMessages());
            String latestToolText = latestToolResponseTextInCurrentTurn(state);
            if (latestToolText.isBlank()) {
                return "";
            }
            if (latestToolText.toLowerCase(Locale.ROOT).contains("no relevant knowledge found")) {
                return "No relevant knowledge snippet was retrieved. Please refine the query keywords.";
            }

            String synthesisInput = """
                    User question:
                    %s

                    Retrieved snippets:
                    %s
                    """.formatted(
                    latestUserQuestion.isBlank() ? "(unknown)" : latestUserQuestion,
                    clip(latestToolText, 3000)
            );

            Prompt prompt = Prompt.builder()
                    .chatOptions(chatOptions)
                    .messages(List.of(new UserMessage(synthesisInput)))
                    .build();

            ChatResponse response = chatClient.prompt(prompt)
                    .system("""
                            You are the final answer composer.
                            Write a direct answer for the user's latest question using ONLY the provided retrieval snippets.
                            Rules:
                            - Focus on the asked question and ignore unrelated snippet parts.
                            - Do not output raw payload markers, metadata, XML/JSON, or tool names.
                            - Do not say you will continue searching/retrieving.
                            - If snippets are insufficient, say what is missing in one short sentence.
                            """)
                    .call()
                    .chatClientResponse()
                    .chatResponse();

            String text = response.getResult().getOutput().getText();
            String cleaned = text == null ? "" : stripProcessText(text);
            if (looksLikeDeferral(cleaned)) {
                return summarizeFromSnippet(latestToolText);
            }
            return cleaned;
        } catch (Exception e) {
            log.warn("[GraphEngine] Failed to synthesize final answer: {}", e.getMessage());
            return "";
        }
    }

    private RouteDecision parseDecision(String text) {
        if (text == null || text.isBlank()) {
            return RouteDecision.invalid();
        }
        Matcher routeMatcher = ROUTE_PATTERN.matcher(text);
        if (!routeMatcher.find()) {
            return RouteDecision.invalid();
        }
        String routeRaw = routeMatcher.group(1).trim().toUpperCase(Locale.ROOT);
        Route route = "WORKER".equals(routeRaw) ? Route.WORKER : Route.FINISH;

        String answer = "";
        Matcher answerMatcher = ANSWER_PATTERN.matcher(text);
        if (answerMatcher.find()) {
            answer = answerMatcher.group(1).trim();
        }
        return new RouteDecision(route, answer);
    }

    private boolean hasRecentToolFailureInCurrentTurn(AgentGraphState state) {
        Object failed = state.getAttributes().get("tool_failed");
        return failed instanceof Boolean b && b;
    }

    private String fallbackFinalAnswer(AgentGraphState state) {
        Object err = state.getAttributes().get("last_tool_error");
        if (err instanceof String s && !s.isBlank()) {
            return "Tool execution failed. Error summary: " + clip(s, 220);
        }

        String latestTool = latestToolResponseTextInCurrentTurn(state);
        if (!latestTool.isBlank()) {
            return clip(latestTool.replace("\\n", "\n").replace("\"", "").trim(), 480);
        }

        return "No valid result was produced in this round. Please refine the question and retry.";
    }

    private String latestToolResponseTextInCurrentTurn(AgentGraphState state) {
        List<Message> messages = state.getMessages();
        if (messages == null || messages.isEmpty()) {
            return "";
        }
        int turnStart = getTurnStartIndex(state, messages);
        for (int i = messages.size() - 1; i >= turnStart; i--) {
            Message message = messages.get(i);
            if (message instanceof ToolResponseMessage toolResponseMessage
                    && toolResponseMessage.getResponses() != null
                    && !toolResponseMessage.getResponses().isEmpty()) {
                return toolResponseMessage.getResponses().stream()
                        .map(ToolResponseMessage.ToolResponse::responseData)
                        .reduce((a, b) -> a + "\n" + b)
                        .orElse("");
            }
        }
        return "";
    }

    private boolean hasLatestToolResponseInCurrentTurn(AgentGraphState state) {
        List<Message> messages = state.getMessages();
        if (messages == null || messages.isEmpty()) {
            return false;
        }
        int turnStart = getTurnStartIndex(state, messages);
        if (messages.size() <= turnStart) {
            return false;
        }
        return messages.get(messages.size() - 1) instanceof ToolResponseMessage;
    }

    private boolean hasToolResponseNamedInCurrentTurn(AgentGraphState state, String toolName) {
        List<Message> messages = state.getMessages();
        if (messages == null || messages.isEmpty() || toolName == null || toolName.isBlank()) {
            return false;
        }
        int turnStart = getTurnStartIndex(state, messages);
        for (int i = messages.size() - 1; i >= turnStart; i--) {
            Message message = messages.get(i);
            if (message instanceof ToolResponseMessage toolResponseMessage && toolResponseMessage.getResponses() != null) {
                for (ToolResponseMessage.ToolResponse response : toolResponseMessage.getResponses()) {
                    if (toolName.equalsIgnoreCase(response.name())) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private boolean shouldForceWorkerByUserIntent(List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return false;
        }
        for (int i = messages.size() - 1; i >= 0; i--) {
            Message message = messages.get(i);
            if (message instanceof UserMessage userMessage) {
                String text = userMessage.getText();
                if (text == null) {
                    return false;
                }
                String lower = text.toLowerCase(Locale.ROOT);
                return lower.contains("weather")
                        || lower.contains("exchange")
                        || lower.contains("sql")
                        || lower.contains("database")
                        || lower.contains("query")
                        || lower.contains("email")
                        || lower.contains("knowledge");
            }
        }
        return false;
    }

    private boolean isWeatherIntent(List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return false;
        }
        for (int i = messages.size() - 1; i >= 0; i--) {
            Message message = messages.get(i);
            if (message instanceof UserMessage userMessage) {
                String text = userMessage.getText();
                if (text == null) {
                    return false;
                }
                String lower = text.toLowerCase(Locale.ROOT);
                return lower.contains("weather");
            }
        }
        return false;
    }

    private int getTurnStartIndex(AgentGraphState state, List<Message> messages) {
        Object value = state.getAttributes().get("turn_start_message_count");
        if (value instanceof Number n) {
            int idx = n.intValue();
            if (idx < 0) {
                return 0;
            }
            return Math.min(idx, messages.size());
        }
        return 0;
    }

    private int getIntAttr(AgentGraphState state, String key, int defaultValue) {
        Object value = state.getAttributes().get(key);
        if (value instanceof Number number) {
            return number.intValue();
        }
        return defaultValue;
    }

    private boolean hasToolName(String expectedToolName) {
        if (expectedToolName == null || expectedToolName.isBlank()) {
            return false;
        }
        if (availableToolNames == null || availableToolNames.isEmpty()) {
            return false;
        }
        for (String toolName : availableToolNames) {
            if (expectedToolName.equalsIgnoreCase(toolName)) {
                return true;
            }
        }
        return false;
    }

    private String getStringAttr(AgentGraphState state, String key) {
        Object value = state.getAttributes().get(key);
        if (value instanceof String text) {
            return text.trim();
        }
        return "";
    }

    private String stripProcessText(String text) {
        if (text == null) {
            return "";
        }
        String cleaned = text.trim();
        cleaned = cleaned.replaceAll("(?i)^(now|next|i will|let me)[^.?!]*[.?!]?", "").trim();
        return cleaned.isBlank() ? text.trim() : cleaned;
    }

    private String clip(String text, int maxLen) {
        if (text == null) {
            return "";
        }
        if (text.length() <= maxLen) {
            return text;
        }
        return text.substring(0, maxLen) + "...";
    }

    private String latestUserQuestionText(List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return "";
        }
        for (int i = messages.size() - 1; i >= 0; i--) {
            Message message = messages.get(i);
            if (message instanceof UserMessage userMessage) {
                return userMessage.getText() == null ? "" : userMessage.getText().trim();
            }
        }
        return "";
    }

    private boolean looksLikeDeferral(String text) {
        if (text == null || text.isBlank()) {
            return false;
        }
        String lower = text.toLowerCase(Locale.ROOT);
        return lower.contains("let me search")
                || lower.contains("i will search")
                || lower.contains("search more")
                || lower.contains("retrieve more")
                || text.contains("\u8ba9\u6211\u518d\u641c\u7d22")
                || text.contains("\u6211\u518d\u68c0\u7d22")
                || text.contains("\u7ee7\u7eed\u68c0\u7d22");
    }

    private String summarizeFromSnippet(String latestToolText) {
        if (latestToolText == null || latestToolText.isBlank()) {
            return "";
        }
        String cleaned = latestToolText
                .replace("\\n", "\n")
                .replace("RETRIEVAL_SNIPPETS", "")
                .replace("Use this section for reasoning and answering.", "")
                .replace("\"", "")
                .trim();
        String[] parts = cleaned.split("\\r?\\n\\s*\\r?\\n");
        for (String part : parts) {
            String p = part.trim();
            if (p.isBlank()) {
                continue;
            }
            if (p.toLowerCase(Locale.ROOT).contains("source_meta_for_display_only")) {
                continue;
            }
            return clip(p, 380);
        }
        return clip(cleaned, 380);
    }

    private enum Route {
        WORKER,
        FINISH,
        INVALID
    }

    private record RouteDecision(Route route, String answer) {
        static RouteDecision invalid() {
            return new RouteDecision(Route.INVALID, "");
        }
    }
}
