package com.kama.jchatmind.agent.graph;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.ChatOptions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
public class SupervisorNode implements AgentNode {

    private static final String RAG_WORKER = "RAG_WORKER";
    private static final String SQL_WORKER = "SQL_WORKER";
    private static final String TOOL_WORKER = "TOOL_WORKER";
    private static final String FINISH = "FINISH";

    private static final String ATTR_PLAN = "task_plan";
    private static final String ATTR_CURRENT_STEP_ID = "task_current_step_id";
    private static final String ATTR_CURRENT_STEP_DESC = "task_current_step_desc";
    private static final String ATTR_CURRENT_STEP_DOMAIN = "task_current_step_domain";
    private static final String ATTR_CURRENT_STEP_POLICY = "task_current_step_policy";
    private static final String ATTR_CURRENT_STEP_ALLOWED_TOOLS = "task_current_step_allowed_tools";
    private static final String ATTR_POLICY_RETRY_FEEDBACK = "policy_retry_feedback";
    private static final String ATTR_LATEST_STEP_EVIDENCE = "latest_step_evidence";
    private static final String ATTR_STEP_EVIDENCE_BY_ID = "step_evidence_by_id";
    private static final double STRICT_DOMAIN_CONFIDENCE = 0.75d;
    private static final int MAX_SAME_DOMAIN_CORRECTIONS = 1;
    private static final int MAX_POLICY_VIOLATIONS = 2;
    private static final int MAX_NORMAL_RETRIES = 2;
    private static final Pattern EMAIL_PATTERN = Pattern.compile("[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}");

    private static final String[] CN_DB = {
            "\u6570\u636e\u5e93", "\u6570\u636e\u8868", "\u8868\u7ed3\u6784", "\u5b57\u6bb5",
            "\u63d2\u5165", "\u65b0\u589e", "\u66f4\u65b0", "\u5220\u9664", "\u67e5\u8be2"
    };
    private static final String[] CN_TOOL = {
            "\u90ae\u4ef6", "\u90ae\u7bb1", "\u53d1\u9001", "\u901a\u77e5", "\u5929\u6c14", "\u57ce\u5e02"
    };
    private static final String[] CN_RAG = {
            "\u77e5\u8bc6", "\u68c0\u7d22", "\u6587\u6863", "\u4ecb\u7ecd", "\u89e3\u91ca",
            "\u8bf4\u660e", "\u603b\u7ed3", "\u5bf9\u6bd4", "\u4ef7\u683c"
    };
    private static final String[] CN_AFTER_ALL = {
            "\u5168\u90e8\u5b8c\u6210", "\u5b8c\u6210\u540e", "\u6700\u540e", "\u4ee5\u4e0a\u5de5\u4f5c"
    };

    // Kept for constructor compatibility with existing wiring.
    @SuppressWarnings("unused")
    private final ChatClient chatClient;
    @SuppressWarnings("unused")
    private final ChatOptions chatOptions;
    @SuppressWarnings("unused")
    private final Set<String> availableToolNames;
    private final Set<String> availableWorkerNames;
    private final ToolPolicyResolver policyResolver;
    private final StepEvaluator stepEvaluator;

    public SupervisorNode(ChatClient chatClient,
                          ChatOptions chatOptions,
                          Set<String> availableToolNames,
                          Set<String> availableWorkerNames) {
        this.chatClient = chatClient;
        this.chatOptions = chatOptions;
        this.availableToolNames = availableToolNames;
        this.availableWorkerNames = availableWorkerNames;
        this.policyResolver = new ToolPolicyResolver();
        this.stepEvaluator = new StepEvaluator(policyResolver);
    }

    @Override
    public String getName() {
        return "SUPERVISOR";
    }

    @Override
    public AgentGraphState process(AgentGraphState state) {
        log.info("[GraphEngine] Node: {}", getName());

        TaskPlan plan = getOrBuildPlan(state);
        if (plan == null) {
            return finishWithAnswer(state, "Unable to build a valid task plan from this request.");
        }
        plan.markRunning();

        handleRunningStepTransition(state, plan);

        while (true) {
            TaskStep next = plan.nextRunnableStep();
            if (next == null) {
                if (plan.allStepsTerminal()) {
                    plan.updateFinalStatus();
                    return finishWithAnswer(state, buildFinalSummary(plan));
                }
                return finishWithAnswer(state, "Workflow paused because no runnable step is available.");
            }

            String candidate = next.currentCandidateWorker();
            if (isBlank(candidate)) {
                markStepFailedIfExecuted(next, "No candidate worker available.");
                continue;
            }

            if (!workerSupportsDomain(next, candidate)) {
                String err = "Worker " + candidate + " is not compatible with domain " + next.getDomain() + ".";
                if (next.moveToNextCandidate()) {
                    next.markPendingForRetry(err);
                    continue;
                }
                if (!tryReclassifyDomain(next, state, err)) {
                    markStepFailedIfExecuted(next, err);
                }
                continue;
            }

            next.markRunning(candidate);
            ToolPolicy policy = resolvePolicy(next);
            log.info("[GraphEngine] Step dispatch: id={}, worker={}, attempt={}, effectiveExecuted={}, desc={}",
                    next.getId(), candidate, next.getAttemptCount(), next.isEffectiveExecuted(), clip(next.getDescription(), 80));
            state.getAttributes().put(ATTR_CURRENT_STEP_ID, next.getId());
            state.getAttributes().put(ATTR_CURRENT_STEP_DESC, next.getDescription());
            state.getAttributes().put(ATTR_CURRENT_STEP_DOMAIN, next.getDomain() == null ? "" : next.getDomain().name());
            state.getAttributes().put(ATTR_CURRENT_STEP_POLICY, policy);
            state.getAttributes().put(ATTR_CURRENT_STEP_ALLOWED_TOOLS, new ArrayList<String>(policy.getAllowedTools()));
            state.getAttributes().put("route_reason", buildRouteReason(next));
            state.getAttributes().put("worker_hops", getIntAttr(state, "worker_hops", 0) + 1);
            state.setNextNode(candidate);
            return state;
        }
    }

    private void handleRunningStepTransition(AgentGraphState state, TaskPlan plan) {
        TaskStep running = getRunningStep(state, plan);
        if (running == null) {
            return;
        }

        StepEvidence evidence = getEvidenceForStep(state, running);
        ToolPolicy policy = resolvePolicy(running, state);
        boolean toolFailed = getBooleanAttr(state, "tool_failed", false);
        String lastToolError = getStringAttr(state, "last_tool_error");
        StepEvaluationResult evaluation = stepEvaluator.evaluate(running, evidence, policy, toolFailed, lastToolError);
        if (evidence != null) {
            evidence.setEvaluatorDecision(evaluation.getDecision().name());
        }
        log.info("[GraphEngine] Step evaluate: id={}, decision={}, policy={}, reason={}",
                running.getId(), evaluation.getDecision(), policy.getPolicyName(), clip(evaluation.getReason(), 160));

        if (evaluation.getDecision() == StepDecision.PASS) {
            running.markEffectiveExecuted();
            running.markSuccess(clip(normalizeToolText(evaluation.getResultSummary()), 800));
            clearCurrentStepContext(state);
            clearExecutionAttrs(state);
            return;
        }

        if (evaluation.getDecision() == StepDecision.RETRY_SAME_WORKER) {
            if (evaluation.isPolicyViolation()) {
                String reason = normalizePolicyViolationReason(evaluation.getReason());
                if (running.canRetryPolicyViolation(MAX_POLICY_VIOLATIONS)) {
                    running.markPolicyViolationForRetry(
                            clip(reason, 200),
                            "Stay on current worker and call only allowed tools: " + String.join(", ", policy.allowedToolList())
                    );
                    state.getAttributes().put(
                            ATTR_POLICY_RETRY_FEEDBACK,
                            "Last attempt violated step policy: " + clip(reason, 220)
                                    + ". You MUST call only: " + String.join(", ", policy.allowedToolList())
                    );
                    clearExecutionAttrs(state);
                    return;
                }
                tryFallbackWorkerOrFail(running, state, "Policy retries exhausted. " + reason);
                if (running.isTerminal()) {
                    clearCurrentStepContext(state);
                    clearExecutionAttrs(state);
                }
                return;
            }
            if (running.getNormalAttemptCount() < MAX_NORMAL_RETRIES) {
                running.markPendingForRetry(clip(evaluation.getReason(), 200));
                clearExecutionAttrs(state);
                return;
            }
            tryFallbackWorkerOrFail(running, state, "Retries exhausted. " + evaluation.getReason());
            if (running.isTerminal()) {
                clearCurrentStepContext(state);
            }
            clearExecutionAttrs(state);
            return;
        }

        if (evaluation.getDecision() == StepDecision.RECLASSIFY) {
            if (!tryReclassifyDomain(running, state, evaluation.getReason())) {
                tryFallbackWorkerOrFail(running, state, evaluation.getReason());
                if (running.isTerminal()) {
                    clearCurrentStepContext(state);
                }
            }
            clearExecutionAttrs(state);
            return;
        }

        markStepFailedIfExecuted(running, evaluation.getReason());
        if (running.isTerminal()) {
            clearCurrentStepContext(state);
        }
        clearExecutionAttrs(state);
    }

    private void tryFallbackWorkerOrFail(TaskStep step, AgentGraphState state, String error) {
        if (step.moveToNextCandidate()) {
            step.markPendingForRetry("Retry with fallback worker. " + clip(error, 200));
            return;
        }
        if (tryReclassifyDomain(step, state, error)) {
            return;
        }
        markStepFailedIfExecuted(step, error);
    }

    private void tryFallbackWithoutFail(TaskStep step, AgentGraphState state, String reason) {
        if (step.moveToNextCandidate()) {
            step.markPendingForRetry("Retry with fallback worker. " + clip(reason, 200));
            return;
        }
        if (tryReclassifyDomain(step, state, reason)) {
            return;
        }
        if (step.getAttemptCount() > 0) {
            step.markSkipped("Unresolved after attempts. " + clip(reason, 200));
        } else {
            step.markSkipped("Not executed. " + clip(reason, 200));
        }
    }

    private void markStepFailedIfExecuted(TaskStep step, String reason) {
        if (step == null) {
            return;
        }
        if (step.isEffectiveExecuted()) {
            step.markFailed(reason);
            return;
        }
        if (step.getAttemptCount() > 0) {
            step.markSkipped("Unresolved after attempts. " + reason);
            return;
        }
        step.markSkipped("Not executed. " + reason);
    }

    private boolean isStepOutputAcceptable(TaskStep step, AgentGraphState state, StepEvidence evidence) {
        if (step == null) {
            return false;
        }
        String descLower = step.getDescription() == null ? "" : step.getDescription().toLowerCase(Locale.ROOT);
        List<String> executedTools = evidence == null ? Collections.<String>emptyList() : evidence.getToolCalls();
        if (executedTools == null || executedTools.isEmpty()) {
            executedTools = getStringListAttr(state, "last_executed_tool_names");
        }
        if (executedTools.isEmpty()) {
            return step.getDomain() == TaskDomain.GENERAL;
        }

        if (step.getDomain() == TaskDomain.SQL) {
            return containsToolName(executedTools, "databaseQuery");
        }
        if (step.getDomain() == TaskDomain.RAG) {
            return containsToolName(executedTools, "KnowledgeTool");
        }
        if (step.getDomain() == TaskDomain.TOOL) {
            if (isEmailIntent(descLower)) {
                return containsAnyTool(executedTools, "sendEmail", "email", "mail");
            }
            if (isWeatherIntent(descLower)) {
                if (containsAnyTool(executedTools, "weather", "getWeather")) {
                    return true;
                }
                return looksLikeWeatherOutput(latestToolResponseTextInCurrentTurn(state));
            }
            return true;
        }
        return true;
    }

    private boolean trySameDomainCorrection(TaskStep step, AgentGraphState state, String reason) {
        if (step == null || state == null) {
            return false;
        }
        if (step.getDomain() != TaskDomain.TOOL) {
            return false;
        }
        if (!step.canRetrySameDomainCorrection(MAX_SAME_DOMAIN_CORRECTIONS)) {
            return false;
        }

        String descLower = step.getDescription() == null ? "" : step.getDescription().toLowerCase(Locale.ROOT);
        List<String> executedTools = getStringListAttr(state, "last_executed_tool_names");

        if (isWeatherIntent(descLower) && !containsAnyTool(executedTools, "weather", "getWeather")) {
            step.markPendingForSameDomainCorrection(
                    "Missed required weather capability. " + clip(reason, 160),
                    "This step is a weather request. You must call a weather tool (weather/getWeather). Date-only tools are insufficient."
            );
            return true;
        }

        if (isEmailIntent(descLower) && !containsAnyTool(executedTools, "sendEmail", "email", "mail")) {
            step.markPendingForSameDomainCorrection(
                    "Missed required email capability. " + clip(reason, 160),
                    "This step is an email action. You must call an email sending tool."
            );
            return true;
        }
        return false;
    }

    private boolean isTextOnlyOutputAcceptable(TaskStep step) {
        if (step == null) {
            return false;
        }
        return step.getDomain() == TaskDomain.GENERAL;
    }

    private boolean containsToolName(List<String> executedTools, String target) {
        if (executedTools == null || target == null) {
            return false;
        }
        for (String tool : executedTools) {
            if (tool != null && target.equalsIgnoreCase(tool.trim())) {
                return true;
            }
        }
        return false;
    }

    private boolean containsAnyTool(List<String> executedTools, String... targets) {
        if (executedTools == null || targets == null) {
            return false;
        }
        for (String tool : executedTools) {
            if (tool == null) {
                continue;
            }
            String t = tool.toLowerCase(Locale.ROOT);
            for (String target : targets) {
                if (target != null && !target.trim().isEmpty() && t.contains(target.toLowerCase(Locale.ROOT))) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean isInvalidAttemptError(String error) {
        if (isBlank(error)) {
            return false;
        }
        String lower = error.toLowerCase(Locale.ROOT);
        return lower.contains("cannot call tool")
                || lower.contains("not compatible with domain")
                || lower.contains("no toolcallback found");
    }

    private boolean isEmailIntent(String descLower) {
        return descLower.contains("email")
                || descLower.contains("mail")
                || descLower.contains("\u90ae\u4ef6")
                || descLower.contains("\u53d1\u9001")
                || descLower.contains("\u90ae\u7bb1");
    }

    private boolean isWeatherIntent(String descLower) {
        return descLower.contains("weather")
                || descLower.contains("\u5929\u6c14")
                || descLower.contains("\u6c14\u6e29");
    }

    private boolean looksLikeWeatherOutput(String text) {
        if (isBlank(text)) {
            return false;
        }
        String lower = text.toLowerCase(Locale.ROOT);
        return containsAny(lower, "weather", "temperature", "humidity", "forecast", "rain", "sunny")
                || containsAny(text, "\u5929\u6c14", "\u6c14\u6e29", "\u6e7f\u5ea6", "\u9884\u62a5", "\u6674", "\u96e8", "\u9634");
    }

    private String buildRouteReason(TaskStep step) {
        ToolPolicy policy = resolvePolicy(step);
        StringBuilder reason = new StringBuilder();
        reason.append("Task: ").append(step.getDescription());
        reason.append(". Domain confidence: ").append(formatConfidence(step.getDomainConfidence())).append(".");
        if (step.isDeliveryStage()) {
            reason.append(". Delivery stage: complete delivery action after prior tasks.");
        }
        if (step.getDomain() == TaskDomain.SQL) {
            reason.append(". Domain: SQL.");
        } else if (step.getDomain() == TaskDomain.RAG) {
            reason.append(". Domain: RAG/knowledge.");
        } else if (step.getDomain() == TaskDomain.TOOL) {
            reason.append(". Domain: operational tools.");
        }
        if (!isBlank(step.getRetryHint())) {
            reason.append(". Retry guidance: ").append(step.getRetryHint());
        }
        if (policy != null) {
            reason.append(". Policy: ").append(policy.getPolicyName());
            if (!policy.getAllowedTools().isEmpty()) {
                reason.append(". Allowed tools: ").append(String.join(", ", policy.allowedToolList()));
            }
        }
        return reason.toString();
    }

    private TaskPlan getOrBuildPlan(AgentGraphState state) {
        Object value = state.getAttributes().get(ATTR_PLAN);
        if (value instanceof TaskPlan) {
            return (TaskPlan) value;
        }
        String userRequest = latestUserQuestionText(state.getMessages());
        TaskPlan plan = buildTaskPlan(userRequest);
        state.getAttributes().put(ATTR_PLAN, plan);
        return plan;
    }

    private TaskPlan buildTaskPlan(String request) {
        List<String> clauses = splitIntoClauses(request);
        List<TaskStep> normalSteps = new ArrayList<TaskStep>();
        List<TaskStep> deliverySteps = new ArrayList<TaskStep>();
        int index = 1;

        for (String clause : clauses) {
            if (isBlank(clause)) {
                continue;
            }
            if (isCoordinatorClause(clause)) {
                continue;
            }
            DomainDecision decision = detectDomainDecision(clause);
            TaskDomain domain = decision.domain;
            boolean delivery = isDeliveryClause(clause);
            List<String> candidates = buildCandidateWorkers(domain, delivery, decision.confidence);
            TaskStep step = new TaskStep("step-" + index, clause.trim(), domain, decision.confidence, delivery, candidates);
            index++;
            if (delivery) {
                deliverySteps.add(step);
            } else {
                normalSteps.add(step);
            }
        }

        List<TaskStep> ordered = new ArrayList<TaskStep>();
        ordered.addAll(normalSteps);
        ordered.addAll(deliverySteps);
        if (ordered.isEmpty()) {
            ordered.add(new TaskStep("step-1", request, TaskDomain.GENERAL, 0.4d, false,
                    buildCandidateWorkers(TaskDomain.GENERAL, false, 0.4d)));
        }
        return new TaskPlan(request, ordered);
    }

    private List<String> splitIntoClauses(String text) {
        if (isBlank(text)) {
            return Collections.emptyList();
        }
        MaskedText masked = maskEmails(text);
        String normalized = masked.maskedText.replace("\u3002", ".")
                .replace("\uff0c", ",")
                .replace("\uff1b", ";")
                .replace("\r", "\n");
        String[] lines = normalized.split("\n+");
        List<String> result = new ArrayList<String>();
        for (String line : lines) {
            if (isBlank(line)) {
                continue;
            }
            String[] pieces = line.split("[,;.]");
            for (String piece : pieces) {
                String trimmed = piece == null ? "" : unmaskEmails(piece.trim(), masked.emailMap);
                if (!isBlank(trimmed)) {
                    result.add(trimmed);
                }
            }
        }
        return mergeDeliveryClauses(result);
    }

    private List<String> mergeDeliveryClauses(List<String> clauses) {
        if (clauses == null || clauses.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> merged = new ArrayList<String>();
        for (String clause : clauses) {
            String c = clause == null ? "" : clause.trim();
            if (isBlank(c)) {
                continue;
            }
            if (!merged.isEmpty() && isLikelyDeliveryTail(c)) {
                int last = merged.size() - 1;
                merged.set(last, merged.get(last) + ", " + c);
            } else {
                merged.add(c);
            }
        }
        return merged;
    }

    private boolean isCoordinatorClause(String clause) {
        String lower = clause.toLowerCase(Locale.ROOT);
        return containsAny(lower, "after all", "once done", "when finished", "finally", "then")
                || containsAny(clause,
                "\u5728\u4ee5\u4e0a\u5de5\u4f5c\u5168\u90e8\u5e72\u5b8c\u4e4b\u540e",
                "\u5728\u4ee5\u4e0a\u5de5\u4f5c\u5168\u90e8\u5b8c\u6210\u4e4b\u540e",
                "\u4ee5\u4e0a\u5de5\u4f5c\u5168\u90e8\u5b8c\u6210\u540e",
                "\u5b8c\u6210\u540e",
                "\u6700\u540e");
    }

    private boolean isLikelyDeliveryTail(String clause) {
        String lower = clause.toLowerCase(Locale.ROOT);
        return containsAny(lower, "send", "email", "mail")
                || containsAny(clause, "\u53d1\u9001", "\u90ae\u4ef6", "\u90ae\u7bb1");
    }

    private MaskedText maskEmails(String text) {
        Map<String, String> emailMap = new HashMap<String, String>();
        Matcher matcher = EMAIL_PATTERN.matcher(text);
        StringBuffer sb = new StringBuffer();
        int idx = 0;
        while (matcher.find()) {
            String email = matcher.group();
            String token = "__EMAIL_" + idx + "__";
            emailMap.put(token, email);
            matcher.appendReplacement(sb, token);
            idx++;
        }
        matcher.appendTail(sb);
        return new MaskedText(sb.toString(), emailMap);
    }

    private String unmaskEmails(String text, Map<String, String> emailMap) {
        if (isBlank(text) || emailMap == null || emailMap.isEmpty()) {
            return text;
        }
        String out = text;
        for (Map.Entry<String, String> entry : emailMap.entrySet()) {
            out = out.replace(entry.getKey(), entry.getValue());
        }
        return out;
    }

    private DomainDecision detectDomainDecision(String clause) {
        String lower = clause.toLowerCase(Locale.ROOT);

        int sqlScore = countKeywordMatches(lower, "database", "sql", "table", "schema", "column", "insert", "update", "delete")
                + countKeywordMatches(clause, CN_DB);
        int toolScore = countKeywordMatches(lower, "email", "mail", "send", "notify", "weather", "city", "api")
                + countKeywordMatches(clause, CN_TOOL);
        int ragScore = countKeywordMatches(lower, "knowledge", "rag", "document", "retrieve", "introduce", "explain", "compare", "price")
                + countKeywordMatches(clause, CN_RAG);

        if (isWeatherIntent(lower) || containsAny(clause, "\u5929\u6c14", "\u6c14\u6e29")) {
            toolScore += 2;
        }
        if (isEmailIntent(lower) || containsAny(clause, "\u90ae\u4ef6", "\u53d1\u9001", "\u90ae\u7bb1")) {
            toolScore += 2;
        }

        int maxScore = Math.max(sqlScore, Math.max(toolScore, ragScore));
        if (maxScore <= 0) {
            return new DomainDecision(TaskDomain.GENERAL, 0.40d);
        }

        int winners = 0;
        winners += sqlScore == maxScore ? 1 : 0;
        winners += toolScore == maxScore ? 1 : 0;
        winners += ragScore == maxScore ? 1 : 0;
        boolean tied = winners > 1;

        TaskDomain topDomain;
        if (toolScore == maxScore && !tied) {
            topDomain = TaskDomain.TOOL;
        } else if (sqlScore == maxScore && !tied) {
            topDomain = TaskDomain.SQL;
        } else if (ragScore == maxScore && !tied) {
            topDomain = TaskDomain.RAG;
        } else if (toolScore == maxScore && (isWeatherIntent(lower) || isEmailIntent(lower))) {
            topDomain = TaskDomain.TOOL;
        } else {
            topDomain = TaskDomain.GENERAL;
        }

        int secondScore = 0;
        if (topDomain == TaskDomain.SQL) {
            secondScore = Math.max(toolScore, ragScore);
        } else if (topDomain == TaskDomain.TOOL) {
            secondScore = Math.max(sqlScore, ragScore);
        } else if (topDomain == TaskDomain.RAG) {
            secondScore = Math.max(toolScore, sqlScore);
        }
        double confidence = calcDomainConfidence(maxScore, secondScore, tied, topDomain);
        return new DomainDecision(topDomain, confidence);
    }

    private boolean isDeliveryClause(String clause) {
        String lower = clause.toLowerCase(Locale.ROOT);
        boolean emailAction = containsAny(lower, "send", "email", "mail")
                || containsAny(clause, "\u53d1\u9001", "\u90ae\u4ef6", "\u90ae\u7bb1");
        if (!emailAction) {
            return false;
        }
        boolean afterAll = containsAny(lower, "after", "once done", "when finished", "finally")
                || containsAny(clause, CN_AFTER_ALL);
        return afterAll;
    }

    private List<String> buildCandidateWorkers(TaskDomain domain, boolean delivery, double domainConfidence) {
        LinkedHashSet<String> ordered = new LinkedHashSet<String>();
        if (delivery) {
            addIfEnabled(ordered, TOOL_WORKER);
            return new ArrayList<String>(ordered);
        }

        if (domain == TaskDomain.SQL) {
            addIfEnabled(ordered, SQL_WORKER);
            if (domainConfidence < STRICT_DOMAIN_CONFIDENCE) {
                addIfEnabled(ordered, RAG_WORKER);
                addIfEnabled(ordered, TOOL_WORKER);
            }
        } else if (domain == TaskDomain.RAG) {
            addIfEnabled(ordered, RAG_WORKER);
            if (domainConfidence < STRICT_DOMAIN_CONFIDENCE) {
                addIfEnabled(ordered, TOOL_WORKER);
                addIfEnabled(ordered, SQL_WORKER);
            }
        } else if (domain == TaskDomain.TOOL) {
            addIfEnabled(ordered, TOOL_WORKER);
            if (domainConfidence < STRICT_DOMAIN_CONFIDENCE) {
                addIfEnabled(ordered, RAG_WORKER);
                addIfEnabled(ordered, SQL_WORKER);
            }
        } else {
            addIfEnabled(ordered, RAG_WORKER);
            addIfEnabled(ordered, SQL_WORKER);
            addIfEnabled(ordered, TOOL_WORKER);
        }
        return new ArrayList<String>(ordered);
    }

    private void addIfEnabled(Set<String> ordered, String worker) {
        if (ordered == null || isBlank(worker)) {
            return;
        }
        if (hasWorker(worker)) {
            ordered.add(worker);
        }
    }

    private boolean workerSupportsDomain(TaskStep step, String worker) {
        if (isBlank(worker) || step == null || step.getDomain() == null) {
            return false;
        }
        TaskDomain domain = step.getDomain();
        String normalized = worker.trim().toUpperCase(Locale.ROOT);
        if (domain == TaskDomain.GENERAL) {
            return true;
        }
        if (!shouldStrictlyEnforceDomain(step)) {
            return true;
        }
        if (domain == TaskDomain.SQL) {
            return SQL_WORKER.equals(normalized);
        }
        if (domain == TaskDomain.RAG) {
            return RAG_WORKER.equals(normalized);
        }
        return TOOL_WORKER.equals(normalized);
    }

    private boolean shouldStrictlyEnforceDomain(TaskStep step) {
        if (step == null) {
            return true;
        }
        if (step.isDeliveryStage()) {
            return true;
        }
        return step.getDomainConfidence() >= STRICT_DOMAIN_CONFIDENCE;
    }

    private boolean tryReclassifyDomain(TaskStep step, AgentGraphState state, String reason) {
        if (step == null || state == null || !step.canReclassifyDomain()) {
            return false;
        }
        DomainDecision decision = inferDomainFromEvidence(step, state, reason);
        if (decision == null || decision.domain == null) {
            return false;
        }
        if (decision.domain == step.getDomain()) {
            return false;
        }
        if (decision.confidence < step.getDomainConfidence() + 0.05d) {
            return false;
        }
        List<String> candidates = buildCandidateWorkers(decision.domain, step.isDeliveryStage(), decision.confidence);
        if (candidates.isEmpty()) {
            return false;
        }
        TaskDomain oldDomain = step.getDomain();
        String msg = "Domain reclassified from " + oldDomain + " to " + decision.domain
                + " (confidence=" + formatConfidence(decision.confidence) + "). "
                + clip(reason, 140);
        step.reclassifyDomain(decision.domain, decision.confidence, candidates, msg);
        log.info("[GraphEngine] Step {} reclassified: {} -> {} (confidence={})",
                step.getId(), oldDomain, decision.domain, formatConfidence(decision.confidence));
        return true;
    }

    private DomainDecision inferDomainFromEvidence(TaskStep step, AgentGraphState state, String reason) {
        StepEvidence evidence = getEvidenceForStep(state, step);
        List<String> tools = evidence == null ? Collections.<String>emptyList() : evidence.getToolCalls();
        if (tools == null || tools.isEmpty()) {
            tools = getStringListAttr(state, "last_executed_tool_names");
        }
        if (containsToolName(tools, "databaseQuery")) {
            return new DomainDecision(TaskDomain.SQL, 0.92d);
        }
        if (containsToolName(tools, "KnowledgeTool")) {
            return new DomainDecision(TaskDomain.RAG, 0.90d);
        }
        if (containsAnyTool(tools, "weather", "getWeather", "sendEmail", "email", "mail")) {
            return new DomainDecision(TaskDomain.TOOL, 0.90d);
        }

        String evidenceText = evidence == null ? "" : evidence.getTextAnswer();
        String evidenceToolText = evidence == null ? "" : String.join(" ", evidence.getToolResults());
        String mergedText = (step.getDescription() == null ? "" : step.getDescription()) + " "
                + (latestToolResponseTextInCurrentTurn(state) == null ? "" : latestToolResponseTextInCurrentTurn(state)) + " "
                + evidenceToolText + " "
                + evidenceText + " "
                + (reason == null ? "" : reason);
        DomainDecision byText = detectDomainDecision(mergedText);
        if (byText.domain == TaskDomain.GENERAL) {
            return null;
        }
        return byText;
    }

    private String buildFinalSummary(TaskPlan plan) {
        int successCount = 0;
        int failedCount = 0;
        int unresolvedCount = 0;
        int skippedCount = 0;
        for (TaskStep step : plan.getSteps()) {
            if (step.getStatus() == TaskStepStatus.SUCCESS) {
                successCount++;
                continue;
            }
            if (step.getStatus() == TaskStepStatus.FAILED) {
                failedCount++;
                continue;
            }
            if (step.getStatus() == TaskStepStatus.SKIPPED) {
                if (step.getNormalAttemptCount() > 0 || step.getPolicyViolationCount() > 0) {
                    unresolvedCount++;
                } else {
                    skippedCount++;
                }
            }
        }
        if (plan.getSteps() == null || plan.getSteps().isEmpty()) {
            return "Workflow finished, but no task output was produced.";
        }

        StringBuilder out = new StringBuilder();
        out.append("任务执行结果概览\n\n")
                .append("本次工作一共包含 ").append(plan.getSteps().size()).append(" 个步骤：\n")
                .append("已成功完成：").append(successCount).append(" 个\n")
                .append("失败：").append(failedCount).append(" 个\n")
                .append("未解决：").append(unresolvedCount).append(" 个\n")
                .append("已跳过：").append(skippedCount).append(" 个\n\n")
                .append("各步骤执行情况\n");

        int index = 1;
        for (TaskStep step : plan.getSteps()) {
            String statusLabel = summarizeStepStatus(step);
            out.append(index).append(". ").append(step.getDescription()).append("\n\n")
                    .append("结果：").append(statusLabel).append("\n");
            if (step.getStatus() == TaskStepStatus.SUCCESS) {
                out.append("输出：").append(formatStepResult(step.getResultSummary())).append("\n");
                String hint = qualityHint(step);
                if (!isBlank(hint)) {
                    out.append("质量提示：").append(hint).append("\n");
                }
            } else {
                out.append("原因：").append(formatFailureReason(step.getErrorSummary())).append("\n");
            }
            if (step.getPolicyViolationCount() > 0) {
                out.append("策略违规次数：").append(step.getPolicyViolationCount()).append("\n");
            }
            if (step.getNormalAttemptCount() > 0) {
                out.append("尝试次数：").append(step.getNormalAttemptCount()).append(" 次\n");
            }
            out.append("\n");
            index++;
        }
        return out.toString().trim();
    }

    private String summarizeStepStatus(TaskStep step) {
        if (step == null) {
            return "未知";
        }
        if (step.getStatus() == TaskStepStatus.SUCCESS) {
            return "成功";
        }
        if (step.getStatus() == TaskStepStatus.FAILED) {
            return "失败";
        }
        if (step.getStatus() == TaskStepStatus.SKIPPED) {
            if (step.getNormalAttemptCount() > 0 || step.getPolicyViolationCount() > 0) {
                return "未解决";
            }
            return "已跳过";
        }
        return step.getStatus().name();
    }

    private String qualityHint(TaskStep step) {
        if (step == null || step.getStatus() != TaskStepStatus.SUCCESS) {
            return "";
        }
        String desc = step.getDescription() == null ? "" : step.getDescription();
        String descLower = desc.toLowerCase(Locale.ROOT);
        String result = step.getResultSummary() == null ? "" : step.getResultSummary();
        String resultLower = result.toLowerCase(Locale.ROOT);

        if (containsAny(descLower, "价格", "price", "报价")) {
            boolean hasPriceSignal = containsAny(resultLower, "price", "价格", "￥", "¥", "元")
                    || result.matches(".*\\d+\\s*(元|rmb|cny|usd).*");
            if (!hasPriceSignal) {
                return "该步骤虽然标记为成功，但返回内容里缺少明确价格信息，建议人工复核。";
            }
        }
        if (containsAny(descLower, "天气", "weather")) {
            boolean hasWeatherSignal = containsAny(resultLower, "weather", "天气", "温度", "湿度", "forecast");
            if (!hasWeatherSignal) {
                return "该步骤虽然标记为成功，但返回内容不像天气结果，建议人工复核。";
            }
        }
        if (containsAny(descLower, "发送", "email", "邮件")) {
            boolean emailSent = containsAny(resultLower, "sendemail", "已提交发送", "sent", "success", "发送");
            if (!emailSent) {
                return "该步骤虽然标记为成功，但没有看到明确发送成功信号，建议人工复核。";
            }
        }
        return "";
    }

    private String formatStepResult(String raw) {
        if (isBlank(raw)) {
            return "(no output)";
        }
        String normalized = normalizeToolText(raw);
        String singleLine = normalized.replace("\r", " ")
                .replace("\n", " ")
                .replaceAll("\\s{2,}", " ")
                .trim();
        return clip(singleLine, 220);
    }

    private String formatFailureReason(String raw) {
        if (isBlank(raw)) {
            return "没有可执行的 worker/tool 能完成该步骤。";
        }
        String lower = raw.toLowerCase(Locale.ROOT);
        if (lower.contains("missed required weather capability")) {
            return "工具选择缺少天气能力（例如只调用了日期工具）。";
        }
        if (lower.contains("missed required email capability")) {
            return "工具选择缺少邮件发送能力。";
        }
        if (lower.contains("output did not satisfy step intent")) {
            return "worker 虽有输出，但没有满足该步骤要求。";
        }
        if (lower.contains("domain reclassified")) {
            return clip(raw.replace("\r", " ").replace("\n", " ").trim(), 220);
        }
        if (lower.contains("not compatible with domain") || lower.contains("cannot call tool")) {
            return "路由回退已耗尽：候选 worker 与所需工具不兼容。";
        }
        if (lower.contains("policy violation")) {
            return "策略校验重试次数已耗尽，仍未得到合规执行。";
        }
        return clip(raw.replace("\r", " ").replace("\n", " ").trim(), 220);
    }

    private AgentGraphState finishWithAnswer(AgentGraphState state, String answer) {
        AssistantMessage finalMsg = new AssistantMessage(answer == null ? "" : answer.trim());
        state.getMessages().add(finalMsg);
        state.getAttributes().put("latest_message", finalMsg);
        state.getAttributes().put("worker_hops", 0);
        clearCurrentStepContext(state);
        state.setNextNode(FINISH);
        return state;
    }

    private void clearCurrentStepContext(AgentGraphState state) {
        if (state == null) {
            return;
        }
        state.getAttributes().remove(ATTR_CURRENT_STEP_ID);
        state.getAttributes().remove(ATTR_CURRENT_STEP_DESC);
        state.getAttributes().remove(ATTR_CURRENT_STEP_DOMAIN);
        state.getAttributes().remove(ATTR_CURRENT_STEP_POLICY);
        state.getAttributes().remove(ATTR_CURRENT_STEP_ALLOWED_TOOLS);
        state.getAttributes().remove(ATTR_POLICY_RETRY_FEEDBACK);
    }

    private void clearExecutionAttrs(AgentGraphState state) {
        if (state == null) {
            return;
        }
        state.getAttributes().put("tool_failed", false);
        state.getAttributes().remove("last_tool_error");
        state.getAttributes().remove("worker_candidate_answer");
        state.getAttributes().remove("last_executed_tool_names");
        state.getAttributes().remove("latest_tool_call_msg");
        state.getAttributes().remove("latest_tool_response_msg");
    }

    private String normalizePolicyViolationReason(String reason) {
        if (reason == null) {
            return "Policy violation.";
        }
        String text = reason.trim();
        String lower = text.toLowerCase(Locale.ROOT);
        if (lower.startsWith("policy violation:")) {
            return text.substring("policy violation:".length()).trim();
        }
        return text;
    }

    private ToolPolicy resolvePolicy(TaskStep step) {
        return policyResolver.resolve(step);
    }

    private ToolPolicy resolvePolicy(TaskStep step, AgentGraphState state) {
        Object policy = state == null ? null : state.getAttributes().get(ATTR_CURRENT_STEP_POLICY);
        if (policy instanceof ToolPolicy) {
            return (ToolPolicy) policy;
        }
        ToolPolicy resolved = resolvePolicy(step);
        if (state != null) {
            state.getAttributes().put(ATTR_CURRENT_STEP_POLICY, resolved);
            state.getAttributes().put(ATTR_CURRENT_STEP_ALLOWED_TOOLS, new ArrayList<String>(resolved.getAllowedTools()));
        }
        return resolved;
    }

    private TaskStep getRunningStep(AgentGraphState state, TaskPlan plan) {
        String stepId = getStringAttr(state, ATTR_CURRENT_STEP_ID);
        if (isBlank(stepId)) {
            return null;
        }
        return plan.findById(stepId);
    }

    @SuppressWarnings("unchecked")
    private StepEvidence getEvidenceForStep(AgentGraphState state, TaskStep step) {
        if (state == null) {
            return null;
        }
        if (step != null) {
            Object byId = state.getAttributes().get(ATTR_STEP_EVIDENCE_BY_ID);
            if (byId instanceof Map<?, ?>) {
                Map<String, StepEvidence> map = (Map<String, StepEvidence>) byId;
                StepEvidence exact = map.get(step.getId());
                if (exact != null) {
                    return exact;
                }
            }
        }
        Object latest = state.getAttributes().get(ATTR_LATEST_STEP_EVIDENCE);
        if (latest instanceof StepEvidence) {
            return (StepEvidence) latest;
        }
        return null;
    }

    private String latestToolResponseTextInCurrentTurn(AgentGraphState state) {
        List<Message> messages = state.getMessages();
        if (messages == null || messages.isEmpty()) {
            return "";
        }
        int turnStart = getTurnStartIndex(state, messages);
        for (int i = messages.size() - 1; i >= turnStart; i--) {
            Message message = messages.get(i);
            if (message instanceof ToolResponseMessage) {
                ToolResponseMessage toolResponse = (ToolResponseMessage) message;
                if (toolResponse.getResponses() == null || toolResponse.getResponses().isEmpty()) {
                    continue;
                }
                StringBuilder joined = new StringBuilder();
                for (ToolResponseMessage.ToolResponse r : toolResponse.getResponses()) {
                    if (joined.length() > 0) {
                        joined.append("\n");
                    }
                    joined.append(r.responseData());
                }
                return joined.toString();
            }
        }
        return "";
    }

    private int getTurnStartIndex(AgentGraphState state, List<Message> messages) {
        Object value = state.getAttributes().get("turn_start_message_count");
        if (value instanceof Number) {
            int idx = ((Number) value).intValue();
            return Math.max(0, Math.min(idx, messages.size()));
        }
        return 0;
    }

    private boolean hasWorker(String workerName) {
        if (isBlank(workerName)) {
            return false;
        }
        if (availableWorkerNames == null || availableWorkerNames.isEmpty()) {
            return TOOL_WORKER.equals(workerName);
        }
        for (String worker : availableWorkerNames) {
            if (workerName.equalsIgnoreCase(worker)) {
                return true;
            }
        }
        return false;
    }

    private String latestUserQuestionText(List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return "";
        }
        for (int i = messages.size() - 1; i >= 0; i--) {
            Message message = messages.get(i);
            if (message instanceof UserMessage) {
                UserMessage user = (UserMessage) message;
                return user.getText() == null ? "" : user.getText().trim();
            }
        }
        return "";
    }

    @SuppressWarnings("unchecked")
    private List<String> getStringListAttr(AgentGraphState state, String key) {
        Object value = state.getAttributes().get(key);
        if (!(value instanceof List)) {
            return Collections.emptyList();
        }
        List<?> raw = (List<?>) value;
        List<String> out = new ArrayList<String>();
        for (Object item : raw) {
            if (item instanceof String) {
                out.add((String) item);
            }
        }
        return out;
    }

    private int getIntAttr(AgentGraphState state, String key, int defaultValue) {
        Object value = state.getAttributes().get(key);
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        return defaultValue;
    }

    private boolean getBooleanAttr(AgentGraphState state, String key, boolean defaultValue) {
        Object value = state.getAttributes().get(key);
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        return defaultValue;
    }

    private String getStringAttr(AgentGraphState state, String key) {
        Object value = state.getAttributes().get(key);
        if (value instanceof String) {
            return ((String) value).trim();
        }
        return "";
    }

    private String normalizeToolText(String text) {
        if (text == null) {
            return "";
        }
        return text.replace("\\n", "\n").replace("\"", "").trim();
    }

    private int countKeywordMatches(String text, String... keywords) {
        if (isBlank(text) || keywords == null || keywords.length == 0) {
            return 0;
        }
        int count = 0;
        for (String keyword : keywords) {
            if (!isBlank(keyword) && text.contains(keyword)) {
                count++;
            }
        }
        return count;
    }

    private double calcDomainConfidence(int topScore, int secondScore, boolean tied, TaskDomain topDomain) {
        if (topDomain == TaskDomain.GENERAL) {
            return 0.52d;
        }
        double confidence = 0.60d + Math.min(0.24d, topScore * 0.08d);
        if (secondScore == 0) {
            confidence += 0.11d;
        } else if (secondScore >= topScore) {
            confidence -= 0.20d;
        } else {
            confidence -= Math.min(0.15d, (secondScore * 1.0d / Math.max(1, topScore)) * 0.15d);
        }
        if (tied) {
            confidence -= 0.12d;
        }
        return Math.max(0.35d, Math.min(0.95d, confidence));
    }

    private String formatConfidence(double confidence) {
        double c = Math.max(0d, Math.min(1d, confidence));
        return String.format(Locale.ROOT, "%.2f", c);
    }

    private boolean containsAny(String text, String... keywords) {
        if (isBlank(text) || keywords == null || keywords.length == 0) {
            return false;
        }
        for (String keyword : keywords) {
            if (!isBlank(keyword) && text.contains(keyword)) {
                return true;
            }
        }
        return false;
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

    private boolean isBlank(String text) {
        return text == null || text.trim().isEmpty();
    }

    private static class DomainDecision {
        private final TaskDomain domain;
        private final double confidence;

        private DomainDecision(TaskDomain domain, double confidence) {
            this.domain = domain;
            this.confidence = Math.max(0d, Math.min(1d, confidence));
        }
    }

    private static class MaskedText {
        private final String maskedText;
        private final Map<String, String> emailMap;

        private MaskedText(String maskedText, Map<String, String> emailMap) {
            this.maskedText = maskedText;
            this.emailMap = emailMap;
        }
    }
}
