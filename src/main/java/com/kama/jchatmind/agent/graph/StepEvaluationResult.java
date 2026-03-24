package com.kama.jchatmind.agent.graph;

public class StepEvaluationResult {

    private final StepDecision decision;
    private final String reason;
    private final String resultSummary;
    private final boolean policyViolation;

    public StepEvaluationResult(StepDecision decision, String reason, String resultSummary, boolean policyViolation) {
        this.decision = decision == null ? StepDecision.FAIL : decision;
        this.reason = reason == null ? "" : reason.trim();
        this.resultSummary = resultSummary == null ? "" : resultSummary.trim();
        this.policyViolation = policyViolation;
    }

    public static StepEvaluationResult pass(String summary) {
        return new StepEvaluationResult(StepDecision.PASS, "Accepted.", summary, false);
    }

    public static StepEvaluationResult retry(String reason, boolean policyViolation) {
        return new StepEvaluationResult(StepDecision.RETRY_SAME_WORKER, reason, "", policyViolation);
    }

    public static StepEvaluationResult reclassify(String reason) {
        return new StepEvaluationResult(StepDecision.RECLASSIFY, reason, "", false);
    }

    public static StepEvaluationResult fail(String reason) {
        return new StepEvaluationResult(StepDecision.FAIL, reason, "", false);
    }

    public StepDecision getDecision() {
        return decision;
    }

    public String getReason() {
        return reason;
    }

    public String getResultSummary() {
        return resultSummary;
    }

    public boolean isPolicyViolation() {
        return policyViolation;
    }
}
