package com.kama.jchatmind.agent.graph;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.model.tool.ToolExecutionResult;
import org.springframework.ai.tool.ToolCallback;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@Slf4j
public abstract class AbstractWorkerNode implements AgentNode {

    private static final Pattern INVOKE_NAME_PATTERN = Pattern.compile("name\\s*=\\s*\"([^\"]+)\"", Pattern.CASE_INSENSITIVE);
    private static final String ATTR_CURRENT_STEP_ID = "task_current_step_id";
    private static final String ATTR_CURRENT_STEP_DESC = "task_current_step_desc";
    private static final String ATTR_CURRENT_STEP_DOMAIN = "task_current_step_domain";
    private static final String ATTR_CURRENT_STEP_ALLOWED_TOOLS = "task_current_step_allowed_tools";
    private static final String ATTR_POLICY_RETRY_FEEDBACK = "policy_retry_feedback";
    private static final String ATTR_LATEST_STEP_EVIDENCE = "latest_step_evidence";
    private static final String ATTR_STEP_EVIDENCE_BY_ID = "step_evidence_by_id";

    private final ChatClient chatClient;
    private final ChatOptions chatOptions;
    private final List<ToolCallback> tools;
    private final ToolCallingManager toolCallingManager;
    private final Set<String> availableToolNames;

    protected AbstractWorkerNode(ChatClient chatClient,
                                 ChatOptions chatOptions,
                                 List<ToolCallback> tools,
                                 Set<String> availableToolNames) {
        this.chatClient = chatClient;
        this.chatOptions = chatOptions;
        this.tools = tools;
        this.availableToolNames = availableToolNames;
        this.toolCallingManager = ToolCallingManager.builder().build();
    }

    @Override
    public AgentGraphState process(AgentGraphState state) {
        log.info("[GraphEngine] Node: {}", getName());

        state.getAttributes().put("tool_failed", false);
        state.getAttributes().remove("last_tool_error");
        state.getAttributes().remove("worker_candidate_answer");
        state.getAttributes().remove("last_executed_tool_names");
        state.getAttributes().put("last_worker", getName());

        List<String> stepAllowedTools = getStringListAttr(state, ATTR_CURRENT_STEP_ALLOWED_TOOLS);
        List<String> effectiveAllowedTools = effectiveAllowedTools(stepAllowedTools);
        String toolNames = effectiveAllowedTools.isEmpty() ? "(none)" : String.join(", ", effectiveAllowedTools);
        String requiredAction = getStringAttr(state, "route_reason");
        String policyRetryFeedback = getStringAttr(state, ATTR_POLICY_RETRY_FEEDBACK);
        StepEvidence evidence = initEvidence(state, requiredAction, effectiveAllowedTools);

        String workerPrompt = String.format(
                "You are a specialized worker node: %s.%n%n"
                        + "Available tool names for THIS STEP only (exact): %s%n%n"
                        + "Current required action: %s%n"
                        + "Policy correction (must follow): %s%n"
                        + "Responsibility boundary:%n"
                        + "%s%n%n"
                        + "Rules:%n"
                        + "1) Call tools when needed.%n"
                        + "2) If no tool is needed, provide plain text only.%n"
                        + "3) Never output DSML/XML/function-call markup in text.%n"
                        + "4) Never invent tool names.%n"
                        + "5) Always prioritize the latest user request over older turns.%n"
                        + "6) For databaseQuery, do not repeat previous failed SQL; generate SQL that matches the latest user request.%n",
                getName(),
                toolNames,
                requiredAction.isEmpty() ? "(none)" : requiredAction,
                policyRetryFeedback.isEmpty() ? "(none)" : policyRetryFeedback,
                responsibilityBoundary()
        );

        ToolCallback[] callbacksForStep = resolveStepToolCallbacks(effectiveAllowedTools);
        log.info("[GraphEngine] {} step tool callbacks: {}", getName(), callbackNames(callbacksForStep));
        Prompt prompt = Prompt.builder()
                .chatOptions(chatOptions)
                .messages(state.getMessages())
                .build();

        ChatResponse response = chatClient.prompt(prompt)
                .system(workerPrompt)
                .toolCallbacks(callbacksForStep)
                .call()
                .chatClientResponse()
                .chatResponse();

        AssistantMessage output = response.getResult().getOutput();
        List<AssistantMessage.ToolCall> toolCalls = output.getToolCalls();
        String outputText = output.getText() == null ? "" : output.getText().trim();

        if (toolCalls == null || toolCalls.isEmpty()) {
            String workerText;
            if (looksLikeToolMarkup(outputText)) {
                String maybeToolName = extractToolName(outputText);
                if (!maybeToolName.isBlank() && (availableToolNames == null || !availableToolNames.contains(maybeToolName))) {
                    workerText = "Detected unregistered tool call '" + maybeToolName + "'. Available tools: " + toolNames + ".";
                } else {
                    workerText = "Detected invalid tool-call markup. Available tools: " + toolNames + ".";
                }
                state.getAttributes().put("tool_failed", true);
                state.getAttributes().put("last_tool_error", workerText);
                evidence.setError(workerText);
                evidence.setConfidence(0.0d);
            } else if (!outputText.isBlank()) {
                workerText = outputText;
                evidence.setTextAnswer(workerText);
                evidence.setConfidence(0.65d);
            } else {
                workerText = "Task processed, but no displayable output was produced.";
                evidence.setTextAnswer(workerText);
                evidence.setConfidence(0.20d);
            }

            state.getAttributes().put("worker_candidate_answer", workerText);
            publishEvidence(state, evidence);
            state.setNextNode("SUPERVISOR");
            return state;
        }

        List<String> toolCallNames = extractToolNames(toolCalls);
        evidence.setToolExecutionAttempted(true);
        evidence.setToolCalls(toolCallNames);
        publishEvidence(state, evidence);

        String stepPolicyViolationTool = findToolOutsideStepPolicy(toolCallNames, effectiveAllowedTools);
        if (!stepPolicyViolationTool.isEmpty()) {
            String error = "Policy violation: tool '" + stepPolicyViolationTool
                    + "' is outside step policy. Step allowed tools: " + toolNames + ".";
            log.warn("[GraphEngine] {}", error);
            state.getAttributes().put("tool_failed", true);
            state.getAttributes().put("last_tool_error", error);
            state.getAttributes().put("worker_candidate_answer", error);
            state.getAttributes().remove("latest_tool_call_msg");
            state.getAttributes().remove("latest_tool_response_msg");
            evidence.setError(error);
            evidence.setPolicyViolation(true);
            evidence.setPolicyCheckResult("forbidden_tool");
            evidence.setToolExecutionSucceeded(false);
            evidence.setConfidence(0.0d);
            publishEvidence(state, evidence);
            state.setNextNode("SUPERVISOR");
            return state;
        }

        String unsupportedToolName = findUnsupportedTool(toolCalls);
        if (!unsupportedToolName.isEmpty()) {
            String error = "Tool execution blocked: worker " + getName()
                    + " cannot call tool '" + unsupportedToolName + "'. Available tools: " + toolNames + ".";
            log.warn("[GraphEngine] {}", error);
            state.getAttributes().put("tool_failed", true);
            state.getAttributes().put("last_tool_error", error);
            state.getAttributes().put("worker_candidate_answer", error);
            state.getAttributes().remove("latest_tool_call_msg");
            state.getAttributes().remove("latest_tool_response_msg");
            evidence.setError(error);
            evidence.setPolicyViolation(true);
            evidence.setPolicyCheckResult("worker_whitelist_violation");
            evidence.setToolExecutionSucceeded(false);
            evidence.setConfidence(0.0d);
            publishEvidence(state, evidence);
            state.setNextNode("SUPERVISOR");
            return state;
        }

        try {
            log.info("[GraphEngine] {} executing tool calls...", getName());
            state.getAttributes().put("last_executed_tool_names", toolCallNames);
            ToolExecutionResult executionResult = toolCallingManager.executeToolCalls(prompt, response);

            List<Message> history = executionResult.conversationHistory();
            state.getMessages().add(output);
            state.getAttributes().put("latest_tool_call_msg", output);

            ToolResponseMessage toolResponseMsg = (ToolResponseMessage) history.get(history.size() - 1);

            state.getMessages().add(toolResponseMsg);
            state.getAttributes().put("latest_tool_response_msg", toolResponseMsg);

            List<String> toolResults = toolResponseMsg.getResponses().stream()
                    .map(r -> r.name() + " -> " + r.responseData())
                    .collect(Collectors.toList());
            evidence.setToolResults(toolResults);
            evidence.setToolExecutionSucceeded(true);
            evidence.setPolicyViolation(false);
            evidence.setPolicyCheckResult("executed");
            evidence.setConfidence(0.90d);
            publishEvidence(state, evidence);

            String logMsg = toolResponseMsg.getResponses().stream()
                    .map(r -> r.name() + " -> " + r.responseData())
                    .collect(Collectors.joining(", "));
            log.info("[GraphEngine] Tool results: {}", logMsg);
        } catch (Exception ex) {
            log.error("[GraphEngine] Tool execution failed", ex);
            String error = "Tool execution failed: " + ex.getMessage();
            state.getAttributes().put("tool_failed", true);
            state.getAttributes().put("last_tool_error", error);
            state.getAttributes().put("worker_candidate_answer", error);
            state.getAttributes().remove("latest_tool_call_msg");
            state.getAttributes().remove("latest_tool_response_msg");
            evidence.setError(error);
            evidence.setToolExecutionSucceeded(false);
            evidence.setPolicyCheckResult("execution_failed");
            evidence.setConfidence(0.0d);
            publishEvidence(state, evidence);
        }

        state.setNextNode("SUPERVISOR");
        return state;
    }

    private StepEvidence initEvidence(AgentGraphState state, String routeReason, List<String> allowedTools) {
        StepEvidence evidence = new StepEvidence();
        evidence.setStepId(getStringAttr(state, ATTR_CURRENT_STEP_ID));
        evidence.setStepDescription(getStringAttr(state, ATTR_CURRENT_STEP_DESC));
        evidence.setRequestedDomain(getStringAttr(state, ATTR_CURRENT_STEP_DOMAIN));
        evidence.setSelectedWorker(getName());
        evidence.setRouteReason(routeReason);
        evidence.setAllowedTools(allowedTools == null ? Collections.<String>emptyList() : allowedTools);
        publishEvidence(state, evidence);
        return evidence;
    }

    private List<String> effectiveAllowedTools(List<String> stepAllowedTools) {
        List<String> workerAllowed = asSortedList(availableToolNames);
        if (stepAllowedTools == null || stepAllowedTools.isEmpty()) {
            return workerAllowed;
        }
        List<String> normalizedStepAllowed = new ArrayList<String>();
        for (String tool : stepAllowedTools) {
            if (tool != null && !tool.trim().isEmpty()) {
                normalizedStepAllowed.add(tool.trim());
            }
        }
        List<String> out = new ArrayList<String>();
        for (String workerTool : workerAllowed) {
            for (String stepTool : normalizedStepAllowed) {
                if (workerTool.equalsIgnoreCase(stepTool)) {
                    out.add(workerTool);
                    break;
                }
            }
        }
        return out;
    }

    private ToolCallback[] resolveStepToolCallbacks(List<String> effectiveAllowedTools) {
        if (tools == null || tools.isEmpty()) {
            return new ToolCallback[0];
        }
        if (effectiveAllowedTools == null || effectiveAllowedTools.isEmpty()) {
            return tools.toArray(new ToolCallback[0]);
        }
        List<ToolCallback> filtered = new ArrayList<ToolCallback>();
        for (ToolCallback callback : tools) {
            String callbackName = callbackName(callback);
            if (callbackName.isEmpty()) {
                continue;
            }
            for (String allowed : effectiveAllowedTools) {
                if (allowed != null && callbackName.equalsIgnoreCase(allowed.trim())) {
                    filtered.add(callback);
                    break;
                }
            }
        }
        if (filtered.isEmpty()) {
            return tools.toArray(new ToolCallback[0]);
        }
        return filtered.toArray(new ToolCallback[0]);
    }

    private String callbackName(ToolCallback callback) {
        if (callback == null) {
            return "";
        }
        try {
            Method getToolDefinition = callback.getClass().getMethod("getToolDefinition");
            Object toolDefinition = getToolDefinition.invoke(callback);
            if (toolDefinition != null) {
                String byNameMethod = invokeStringNoArg(toolDefinition, "name");
                if (!byNameMethod.isEmpty()) {
                    return byNameMethod;
                }
                String byGetter = invokeStringNoArg(toolDefinition, "getName");
                if (!byGetter.isEmpty()) {
                    return byGetter;
                }
            }
        } catch (Exception ignored) {
            // Fallback to other reflective access paths.
        }
        String directName = invokeStringNoArg(callback, "getName");
        if (!directName.isEmpty()) {
            return directName;
        }
        return "";
    }

    private String callbackNames(ToolCallback[] callbacks) {
        if (callbacks == null || callbacks.length == 0) {
            return "(none)";
        }
        List<String> names = new ArrayList<String>();
        for (ToolCallback callback : callbacks) {
            String name = callbackName(callback);
            if (!name.isEmpty()) {
                names.add(name);
            }
        }
        if (names.isEmpty()) {
            return "(unresolved)";
        }
        return String.join(", ", names);
    }

    private String invokeStringNoArg(Object target, String methodName) {
        if (target == null || methodName == null || methodName.trim().isEmpty()) {
            return "";
        }
        try {
            Method method = target.getClass().getMethod(methodName);
            Object value = method.invoke(target);
            if (value instanceof String) {
                return ((String) value).trim();
            }
        } catch (Exception ignored) {
            // No-op.
        }
        return "";
    }

    private List<String> asSortedList(Set<String> values) {
        if (values == null || values.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<String>();
        for (String value : values) {
            if (value != null && !value.trim().isEmpty()) {
                out.add(value.trim());
            }
        }
        Collections.sort(out);
        return out;
    }

    @SuppressWarnings("unchecked")
    private void publishEvidence(AgentGraphState state, StepEvidence evidence) {
        if (state == null || evidence == null) {
            return;
        }
        state.getAttributes().put(ATTR_LATEST_STEP_EVIDENCE, evidence);
        Object raw = state.getAttributes().get(ATTR_STEP_EVIDENCE_BY_ID);
        Map<String, StepEvidence> evidenceById;
        if (raw instanceof Map<?, ?>) {
            evidenceById = (Map<String, StepEvidence>) raw;
        } else {
            evidenceById = new HashMap<String, StepEvidence>();
            state.getAttributes().put(ATTR_STEP_EVIDENCE_BY_ID, evidenceById);
        }
        String stepId = evidence.getStepId();
        if (stepId != null && !stepId.isBlank()) {
            evidenceById.put(stepId, evidence);
        }
    }

    private List<String> extractToolNames(List<AssistantMessage.ToolCall> toolCalls) {
        return toolCalls.stream()
                .map(AssistantMessage.ToolCall::name)
                .filter(name -> name != null && !name.trim().isEmpty())
                .collect(Collectors.toList());
    }

    private String findToolOutsideStepPolicy(List<String> toolCallNames, List<String> stepAllowedTools) {
        if (toolCallNames == null || toolCallNames.isEmpty()) {
            return "";
        }
        if (stepAllowedTools == null || stepAllowedTools.isEmpty()) {
            return "";
        }
        for (String call : toolCallNames) {
            boolean matched = false;
            for (String allowed : stepAllowedTools) {
                if (call != null && allowed != null && call.equalsIgnoreCase(allowed)) {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                return call == null ? "" : call.trim();
            }
        }
        return "";
    }

    private String findUnsupportedTool(List<AssistantMessage.ToolCall> toolCalls) {
        if (toolCalls == null || toolCalls.isEmpty()) {
            return "";
        }
        for (AssistantMessage.ToolCall call : toolCalls) {
            if (call == null) {
                continue;
            }
            String name = call.name() == null ? "" : call.name().trim();
            if (name.isEmpty()) {
                return "(empty)";
            }
            if (!supportsToolName(name)) {
                return name;
            }
        }
        return "";
    }

    private boolean supportsToolName(String toolName) {
        if (toolName == null || toolName.trim().isEmpty()) {
            return false;
        }
        if (availableToolNames == null || availableToolNames.isEmpty()) {
            return false;
        }
        for (String name : availableToolNames) {
            if (name != null && toolName.equalsIgnoreCase(name.trim())) {
                return true;
            }
        }
        return false;
    }

    private boolean looksLikeToolMarkup(String text) {
        if (text == null || text.isBlank()) {
            return false;
        }
        String lower = text.toLowerCase(Locale.ROOT);
        return lower.contains("dsml")
                || lower.contains("function_calls")
                || lower.contains("tool_calls")
                || lower.contains("<invoke")
                || lower.contains("<function")
                || lower.contains("</function");
    }

    private String extractToolName(String text) {
        if (text == null || text.isBlank()) {
            return "";
        }
        Matcher matcher = INVOKE_NAME_PATTERN.matcher(text);
        if (matcher.find()) {
            return matcher.group(1).trim();
        }
        return "";
    }

    private String getStringAttr(AgentGraphState state, String key) {
        if (state == null || key == null) {
            return "";
        }
        Object value = state.getAttributes().get(key);
        if (value instanceof String) {
            return ((String) value).trim();
        }
        return "";
    }

    @SuppressWarnings("unchecked")
    private List<String> getStringListAttr(AgentGraphState state, String key) {
        if (state == null || key == null) {
            return Collections.emptyList();
        }
        Object value = state.getAttributes().get(key);
        if (!(value instanceof List<?>)) {
            return Collections.emptyList();
        }
        List<?> raw = (List<?>) value;
        List<String> out = new ArrayList<String>();
        for (Object item : raw) {
            if (item instanceof String) {
                out.add(((String) item).trim());
            }
        }
        return out;
    }

    private String responsibilityBoundary() {
        String name = getName() == null ? "" : getName().toUpperCase(Locale.ROOT);
        if ("RAG_WORKER".equals(name)) {
            return "- Responsible: retrieval, knowledge lookup, context summarization.\n"
                    + "- Not responsible: SQL/database analysis, operational actions (email/file/etc).";
        }
        if ("SQL_WORKER".equals(name)) {
            return "- Responsible: SQL/database/table/metric queries and analysis.\n"
                    + "- Not responsible: retrieval/knowledge search, operational actions.";
        }
        if ("TOOL_WORKER".equals(name)) {
            return "- Responsible: operational tool actions (email/file/weather/external actions).\n"
                    + "- Not responsible: SQL generation/analysis, retrieval reasoning.";
        }
        return "- Stay within the tools and domain of this worker.";
    }
}
