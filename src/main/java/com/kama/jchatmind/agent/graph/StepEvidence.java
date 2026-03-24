package com.kama.jchatmind.agent.graph;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class StepEvidence {

    private String stepId;
    private String stepDescription;
    private String requestedDomain;
    private String selectedWorker;
    private String routeReason;
    private final List<String> allowedTools = new ArrayList<String>();
    private final List<String> toolCalls = new ArrayList<String>();
    private final List<String> toolResults = new ArrayList<String>();
    private String textAnswer;
    private String error;
    private double confidence;
    private boolean toolExecutionAttempted;
    private boolean toolExecutionSucceeded;
    private final long createdAtEpochMs;
    private long updatedAtEpochMs;

    public StepEvidence() {
        long now = System.currentTimeMillis();
        this.createdAtEpochMs = now;
        this.updatedAtEpochMs = now;
        this.confidence = 0.0d;
        this.textAnswer = "";
        this.error = "";
    }

    public String getStepId() {
        return stepId;
    }

    public void setStepId(String stepId) {
        this.stepId = safe(stepId);
        touch();
    }

    public String getStepDescription() {
        return stepDescription;
    }

    public void setStepDescription(String stepDescription) {
        this.stepDescription = safe(stepDescription);
        touch();
    }

    public String getRequestedDomain() {
        return requestedDomain;
    }

    public void setRequestedDomain(String requestedDomain) {
        this.requestedDomain = safe(requestedDomain);
        touch();
    }

    public String getSelectedWorker() {
        return selectedWorker;
    }

    public void setSelectedWorker(String selectedWorker) {
        this.selectedWorker = safe(selectedWorker);
        touch();
    }

    public String getRouteReason() {
        return routeReason;
    }

    public void setRouteReason(String routeReason) {
        this.routeReason = safe(routeReason);
        touch();
    }

    public List<String> getAllowedTools() {
        return Collections.unmodifiableList(allowedTools);
    }

    public void addAllowedTool(String toolName) {
        if (toolName == null || toolName.trim().isEmpty()) {
            return;
        }
        this.allowedTools.add(toolName.trim());
        touch();
    }

    public void setAllowedTools(List<String> toolNames) {
        this.allowedTools.clear();
        if (toolNames != null) {
            for (String toolName : toolNames) {
                addAllowedTool(toolName);
            }
        }
        touch();
    }

    public List<String> getToolCalls() {
        return Collections.unmodifiableList(toolCalls);
    }

    public void addToolCall(String toolName) {
        if (toolName == null || toolName.trim().isEmpty()) {
            return;
        }
        this.toolCalls.add(toolName.trim());
        touch();
    }

    public void setToolCalls(List<String> toolNames) {
        this.toolCalls.clear();
        if (toolNames != null) {
            for (String toolName : toolNames) {
                addToolCall(toolName);
            }
        }
        touch();
    }

    public List<String> getToolResults() {
        return Collections.unmodifiableList(toolResults);
    }

    public void addToolResult(String toolResult) {
        if (toolResult == null || toolResult.trim().isEmpty()) {
            return;
        }
        this.toolResults.add(toolResult.trim());
        touch();
    }

    public void setToolResults(List<String> results) {
        this.toolResults.clear();
        if (results != null) {
            for (String result : results) {
                addToolResult(result);
            }
        }
        touch();
    }

    public String getTextAnswer() {
        return textAnswer;
    }

    public void setTextAnswer(String textAnswer) {
        this.textAnswer = safe(textAnswer);
        touch();
    }

    public String getError() {
        return error;
    }

    public void setError(String error) {
        this.error = safe(error);
        touch();
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = Math.max(0d, Math.min(1d, confidence));
        touch();
    }

    public boolean isToolExecutionAttempted() {
        return toolExecutionAttempted;
    }

    public void setToolExecutionAttempted(boolean toolExecutionAttempted) {
        this.toolExecutionAttempted = toolExecutionAttempted;
        touch();
    }

    public boolean isToolExecutionSucceeded() {
        return toolExecutionSucceeded;
    }

    public void setToolExecutionSucceeded(boolean toolExecutionSucceeded) {
        this.toolExecutionSucceeded = toolExecutionSucceeded;
        touch();
    }

    public long getCreatedAtEpochMs() {
        return createdAtEpochMs;
    }

    public long getUpdatedAtEpochMs() {
        return updatedAtEpochMs;
    }

    private void touch() {
        this.updatedAtEpochMs = System.currentTimeMillis();
    }

    private String safe(String value) {
        return value == null ? "" : value.trim();
    }
}
