package com.kama.jchatmind.agent.graph;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public class StepEvaluator {

    private final ToolPolicyResolver policyResolver;

    public StepEvaluator(ToolPolicyResolver policyResolver) {
        this.policyResolver = policyResolver == null ? new ToolPolicyResolver() : policyResolver;
    }

    public StepEvaluationResult evaluate(TaskStep step,
                                         StepEvidence evidence,
                                         ToolPolicy policy,
                                         boolean toolFailed,
                                         String lastToolError) {
        if (step == null) {
            return StepEvaluationResult.fail("Step is null.");
        }
        if (policy == null) {
            policy = ToolPolicy.builder("GENERAL").build();
        }
        if (evidence == null) {
            return StepEvaluationResult.retry("Missing step evidence.", false);
        }

        List<String> calls = safeList(evidence.getToolCalls());
        List<String> results = safeList(evidence.getToolResults());
        String mergedOutput = (String.join(" ", results) + " " + safe(evidence.getTextAnswer())).trim();

        boolean forbiddenTriggered = containsForbidden(calls, policy);
        boolean requiredToolsSatisfied = containsRequiredTools(calls, policy);
        Set<Capability> observedCapabilities = observedCapabilities(calls);
        boolean requiredCapabilitiesSatisfied = observedCapabilities.containsAll(policy.getRequiredCapabilities());
        boolean requiredEvidenceSatisfied = requiredEvidenceSatisfied(policy, mergedOutput);
        boolean sideEffectObserved = requiredSideEffectsObserved(policy, mergedOutput, calls);

        evidence.setForbiddenTriggered(forbiddenTriggered);
        evidence.setRequiredSatisfied(requiredToolsSatisfied && requiredCapabilitiesSatisfied && requiredEvidenceSatisfied);
        evidence.setSideEffectObserved(sideEffectObserved || policy.getRequiredSideEffects().isEmpty());

        if (toolFailed) {
            String error = safe(lastToolError);
            if (error.isEmpty()) {
                error = safe(evidence.getError());
            }
            if (isPolicyViolationError(error)) {
                evidence.setPolicyViolation(true);
                evidence.setPolicyCheckResult("policy_violation");
                return StepEvaluationResult.retry(error, true);
            }
            if (looksLikeDomainMismatch(error)) {
                evidence.setPolicyCheckResult("domain_mismatch");
                return StepEvaluationResult.reclassify(error);
            }
            evidence.setPolicyCheckResult("tool_failed");
            return StepEvaluationResult.retry(error.isEmpty() ? "Tool execution failed." : error, false);
        }

        if (forbiddenTriggered) {
            evidence.setPolicyViolation(true);
            evidence.setPolicyCheckResult("forbidden_tool");
            return StepEvaluationResult.retry("Forbidden tool used for this step.", true);
        }
        if (!requiredToolsSatisfied) {
            evidence.setPolicyViolation(true);
            evidence.setPolicyCheckResult("required_tool_missing");
            return StepEvaluationResult.retry("Required tool was not called.", true);
        }
        if (!requiredCapabilitiesSatisfied) {
            evidence.setPolicyViolation(true);
            evidence.setPolicyCheckResult("required_capability_missing");
            return StepEvaluationResult.retry("Required capability was not satisfied.", true);
        }
        if (!requiredEvidenceSatisfied) {
            evidence.setPolicyViolation(true);
            evidence.setPolicyCheckResult("required_evidence_missing");
            return StepEvaluationResult.retry("Required evidence missing in output.", true);
        }
        if (!sideEffectObserved) {
            evidence.setPolicyViolation(true);
            evidence.setPolicyCheckResult("required_side_effect_missing");
            return StepEvaluationResult.retry("Required side effect not observed.", true);
        }

        if (policy.isRequireToolExecution() && calls.isEmpty()) {
            evidence.setPolicyViolation(true);
            evidence.setPolicyCheckResult("tool_required_but_not_called");
            return StepEvaluationResult.retry("This step requires tool execution.", true);
        }

        if (!calls.isEmpty() && results.isEmpty() && safe(evidence.getError()).isEmpty()) {
            evidence.setPolicyCheckResult("missing_tool_result");
            return StepEvaluationResult.retry("Tool call has no result.", false);
        }

        if (step.getDomain() == TaskDomain.SQL && looksLikeSqlExecutionFailure(mergedOutput)) {
            evidence.setPolicyCheckResult("sql_execution_failed");
            return StepEvaluationResult.retry("SQL execution failed: " + firstLine(mergedOutput), false);
        }

        if (calls.isEmpty() && safe(evidence.getTextAnswer()).isEmpty()) {
            evidence.setPolicyCheckResult("empty_output");
            return StepEvaluationResult.retry("No usable output.", false);
        }

        evidence.setPolicyCheckResult("pass");
        String summary = !results.isEmpty() ? String.join("\n", results) : safe(evidence.getTextAnswer());
        return StepEvaluationResult.pass(summary);
    }

    private boolean requiredEvidenceSatisfied(ToolPolicy policy, String mergedOutput) {
        if (policy.getRequiredEvidence().isEmpty()) {
            return true;
        }
        String lower = safe(mergedOutput).toLowerCase(Locale.ROOT);
        for (String key : policy.getRequiredEvidence()) {
            if ("weather_result".equalsIgnoreCase(key)) {
                if (containsAny(lower, "weather", "temperature", "humidity", "\u5929\u6c14", "\u6c14\u6e29", "\u6e7f\u5ea6")) {
                    return true;
                }
                continue;
            }
            if (lower.contains(key.toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

    private boolean requiredSideEffectsObserved(ToolPolicy policy, String mergedOutput, List<String> calls) {
        if (policy.getRequiredSideEffects().isEmpty()) {
            return true;
        }
        String lower = safe(mergedOutput).toLowerCase(Locale.ROOT);
        for (String sideEffect : policy.getRequiredSideEffects()) {
            if ("email_sent".equalsIgnoreCase(sideEffect)) {
                if (containsToolName(calls, "sendemail")
                        && containsAny(lower, "submitted", "sent", "success", "\u53d1\u9001", "\u5df2\u63d0\u4ea4")) {
                    return true;
                }
                continue;
            }
            if (lower.contains(sideEffect.toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

    private boolean containsForbidden(List<String> calls, ToolPolicy policy) {
        if (calls == null || calls.isEmpty() || policy.getForbiddenTools().isEmpty()) {
            return false;
        }
        for (String call : calls) {
            if (policy.isForbiddenTool(call)) {
                return true;
            }
        }
        return false;
    }

    private boolean containsRequiredTools(List<String> calls, ToolPolicy policy) {
        if (policy.getRequiredTools().isEmpty()) {
            return true;
        }
        if (calls == null || calls.isEmpty()) {
            return false;
        }
        for (String required : policy.getRequiredTools()) {
            if (!containsToolName(calls, required)) {
                return false;
            }
        }
        return true;
    }

    private Set<Capability> observedCapabilities(List<String> calls) {
        if (calls == null || calls.isEmpty()) {
            return Collections.emptySet();
        }
        Set<Capability> out = new LinkedHashSet<Capability>();
        for (String call : calls) {
            out.addAll(policyResolver.capabilitiesForTool(call));
        }
        return out;
    }

    private boolean containsToolName(List<String> calls, String tool) {
        if (calls == null || tool == null || tool.trim().isEmpty()) {
            return false;
        }
        String expected = tool.trim().toLowerCase(Locale.ROOT);
        for (String call : calls) {
            if (call != null && expected.equals(call.trim().toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

    private boolean looksLikeDomainMismatch(String error) {
        String lower = safe(error).toLowerCase(Locale.ROOT);
        return lower.contains("not compatible with domain")
                || lower.contains("domain mismatch");
    }

    private boolean isPolicyViolationError(String error) {
        String lower = safe(error).toLowerCase(Locale.ROOT);
        return lower.contains("policy violation")
                || lower.contains("cannot call tool")
                || lower.contains("forbidden tool")
                || lower.contains("outside step policy")
                || lower.contains("required tool");
    }

    private boolean containsAny(String text, String... keywords) {
        if (text == null || text.trim().isEmpty() || keywords == null) {
            return false;
        }
        for (String keyword : keywords) {
            if (keyword != null && !keyword.trim().isEmpty() && text.contains(keyword.toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

    private boolean looksLikeSqlExecutionFailure(String output) {
        String lower = safe(output).toLowerCase(Locale.ROOT);
        if (lower.isEmpty()) {
            return false;
        }
        return lower.contains("sql execution failed")
                || lower.contains("sql syntax error")
                || lower.contains("does not exist")
                || lower.contains("not exist")
                || lower.contains("only select")
                || lower.contains("query failed");
    }

    private String firstLine(String text) {
        String safe = safe(text);
        if (safe.isEmpty()) {
            return "unknown";
        }
        int idx = safe.indexOf('\n');
        if (idx < 0) {
            return safe;
        }
        return safe.substring(0, idx).trim();
    }

    private String safe(String value) {
        return value == null ? "" : value.trim();
    }

    private List<String> safeList(List<String> values) {
        return values == null ? Collections.<String>emptyList() : values;
    }
}
