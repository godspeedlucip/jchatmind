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
                - Final answer must be user-facing result only. No planning/process narration.
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

        // 1) First-step intent forcing: only once, avoid loop.
        if (stepCount <= 1 && shouldForceWorkerByUserIntent(messages) && !hasToolFailure) {
            state.getAttributes().put("worker_hops", workerHops + 1);
            state.setNextNode("WORKER");
            return state;
        }

        // 2) Weather guard: if weather intent and weather tool not yet executed in this turn, keep working.
        if (weatherIntent && !hasWeatherResult && workerHops < 6) {
            state.getAttributes().put("worker_hops", workerHops + 1);
            state.setNextNode("WORKER");
            return state;
        }

        // 3) Decision handling.
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
            String finalAnswer = decision.answer;
            if (finalAnswer.isBlank()) {
                finalAnswer = !workerCandidateAnswer.isBlank() ? workerCandidateAnswer : fallbackFinalAnswer(state);
            }
            AssistantMessage finalMsg = new AssistantMessage(stripProcessText(finalAnswer));
            state.getMessages().add(finalMsg);
            state.getAttributes().put("latest_message", finalMsg);
            state.getAttributes().put("worker_hops", 0);
            state.setNextNode("FINISH");
            return state;
        }

        // 4) Invalid decision fallback.
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
            // If tool has run in this turn and model gives no valid route, finish with concise summary.
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
        if (failed instanceof Boolean b && b) {
            return true;
        }

        List<Message> messages = state.getMessages();
        if (messages == null || messages.isEmpty()) {
            return false;
        }
        int turnStart = getTurnStartIndex(state, messages);

        for (int i = messages.size() - 1; i >= turnStart; i--) {
            Message message = messages.get(i);
            if (message instanceof ToolResponseMessage toolResponseMessage && toolResponseMessage.getResponses() != null) {
                for (ToolResponseMessage.ToolResponse response : toolResponseMessage.getResponses()) {
                    if (isFailureText(response.responseData())) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private String fallbackFinalAnswer(AgentGraphState state) {
        Object err = state.getAttributes().get("last_tool_error");
        if (err instanceof String s && !s.isBlank()) {
            return "工具执行失败，暂时无法完成请求。错误摘要: " + clip(s, 220) + "。";
        }

        String latestTool = latestToolResponseTextInCurrentTurn(state);
        if (!latestTool.isBlank()) {
            if (isFailureText(latestTool)) {
                return "工具执行失败，暂时无法完成请求。错误摘要: " + clip(latestTool, 220) + "。";
            }
            return stripProcessText(latestTool);
        }

        return "本轮未获得有效结果，请补充更具体条件后重试。";
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
                        || lower.contains("天气")
                        || lower.contains("汇率")
                        || lower.contains("exchange")
                        || lower.contains("sql")
                        || lower.contains("数据库")
                        || lower.contains("查询")
                        || lower.contains("邮件")
                        || lower.contains("email")
                        || lower.contains("知识库");
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
                return lower.contains("weather") || lower.contains("天气");
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

    private String getStringAttr(AgentGraphState state, String key) {
        Object value = state.getAttributes().get(key);
        if (value instanceof String text) {
            return text.trim();
        }
        return "";
    }

    private boolean isFailureText(String text) {
        if (text == null || text.isBlank()) {
            return false;
        }
        String lower = text.toLowerCase(Locale.ROOT);
        return lower.contains("error")
                || lower.contains("exception")
                || lower.contains("failed")
                || lower.contains("timeout")
                || lower.contains("not exist")
                || lower.contains("invalid")
                || lower.contains("失败")
                || lower.contains("错误")
                || lower.contains("不存在");
    }

    private String stripProcessText(String text) {
        if (text == null) {
            return "";
        }
        String cleaned = text.trim();
        cleaned = cleaned.replaceAll("(?i)^(现在|接下来|我将|我会|让我)[^。！？!?.]*[。！？!?.]?", "").trim();
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
