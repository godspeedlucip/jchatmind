package com.kama.jchatmind.agent.graph;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TaskStep {

    private final String id;
    private final String description;
    private TaskDomain domain;
    private double domainConfidence;
    private final boolean deliveryStage;
    private final List<String> candidateWorkers;
    private int candidateIndex;
    private TaskStepStatus status;
    private String assignedWorker;
    private String resultSummary;
    private String errorSummary;
    private boolean executed;
    private boolean effectiveExecuted;
    private int attemptCount;
    private int sameDomainCorrectionCount;
    private boolean domainReclassified;
    private String retryHint;
    private String transitionReason;
    private final List<String> transitionHistory;

    public TaskStep(String id,
                    String description,
                    TaskDomain domain,
                    boolean deliveryStage,
                    List<String> candidateWorkers) {
        this(id, description, domain, 0.9d, deliveryStage, candidateWorkers);
    }

    public TaskStep(String id,
                    String description,
                    TaskDomain domain,
                    double domainConfidence,
                    boolean deliveryStage,
                    List<String> candidateWorkers) {
        this.id = id;
        this.description = description;
        this.domain = domain;
        this.domainConfidence = Math.max(0d, Math.min(1d, domainConfidence));
        this.deliveryStage = deliveryStage;
        this.candidateWorkers = candidateWorkers == null ? new ArrayList<String>() : new ArrayList<String>(candidateWorkers);
        this.candidateIndex = 0;
        this.status = TaskStepStatus.PENDING;
        this.assignedWorker = "";
        this.resultSummary = "";
        this.errorSummary = "";
        this.executed = false;
        this.effectiveExecuted = false;
        this.attemptCount = 0;
        this.sameDomainCorrectionCount = 0;
        this.domainReclassified = false;
        this.retryHint = "";
        this.transitionReason = "";
        this.transitionHistory = new ArrayList<String>();
        this.transitionHistory.add("INIT -> PENDING: Step created.");
    }

    public String getId() {
        return id;
    }

    public String getDescription() {
        return description;
    }

    public TaskDomain getDomain() {
        return domain;
    }

    public double getDomainConfidence() {
        return domainConfidence;
    }

    public boolean isDeliveryStage() {
        return deliveryStage;
    }

    public List<String> getCandidateWorkers() {
        return Collections.unmodifiableList(candidateWorkers);
    }

    public TaskStepStatus getStatus() {
        return status;
    }

    public String getAssignedWorker() {
        return assignedWorker;
    }

    public String getResultSummary() {
        return resultSummary;
    }

    public String getErrorSummary() {
        return errorSummary;
    }

    public boolean isExecuted() {
        return executed;
    }

    public int getAttemptCount() {
        return attemptCount;
    }

    public int getSameDomainCorrectionCount() {
        return sameDomainCorrectionCount;
    }

    public boolean isDomainReclassified() {
        return domainReclassified;
    }

    public String getRetryHint() {
        return retryHint;
    }

    public boolean isEffectiveExecuted() {
        return effectiveExecuted;
    }

    public boolean isTerminal() {
        return status == TaskStepStatus.SUCCESS || status == TaskStepStatus.FAILED || status == TaskStepStatus.SKIPPED;
    }

    public boolean isRunnable() {
        return status == TaskStepStatus.PENDING
                || status == TaskStepStatus.RETRY_PENDING
                || status == TaskStepStatus.RECLASSIFY_PENDING;
    }

    public String getTransitionReason() {
        return transitionReason;
    }

    public List<String> getTransitionHistory() {
        return Collections.unmodifiableList(transitionHistory);
    }

    public String currentCandidateWorker() {
        if (candidateWorkers.isEmpty()) {
            return "";
        }
        int idx = Math.max(0, Math.min(candidateIndex, candidateWorkers.size() - 1));
        return candidateWorkers.get(idx);
    }

    public boolean moveToNextCandidate() {
        if (candidateWorkers.isEmpty()) {
            return false;
        }
        if (candidateIndex + 1 >= candidateWorkers.size()) {
            return false;
        }
        candidateIndex++;
        return true;
    }

    public void markRunning(String worker) {
        this.assignedWorker = worker == null ? "" : worker;
        this.executed = true;
        this.attemptCount++;
        this.retryHint = "";
        transitionTo(TaskStepStatus.RUNNING, "Dispatched to worker: " + this.assignedWorker);
    }

    public void markEffectiveExecuted() {
        this.effectiveExecuted = true;
    }

    public void markPendingForRetry(String reason) {
        this.errorSummary = reason == null ? "" : reason;
        this.retryHint = "";
        transitionTo(TaskStepStatus.RETRY_PENDING, this.errorSummary);
    }

    public void markPendingForSameDomainCorrection(String reason, String hint) {
        this.errorSummary = reason == null ? "" : reason;
        this.retryHint = hint == null ? "" : hint;
        this.sameDomainCorrectionCount++;
        transitionTo(TaskStepStatus.RETRY_PENDING, this.errorSummary);
    }

    public boolean canRetrySameDomainCorrection(int maxCorrections) {
        return sameDomainCorrectionCount < Math.max(0, maxCorrections);
    }

    public boolean canReclassifyDomain() {
        return !domainReclassified;
    }

    public void reclassifyDomain(TaskDomain newDomain,
                                 double newConfidence,
                                 List<String> newCandidates,
                                 String reason) {
        if (newDomain == null) {
            return;
        }
        this.domain = newDomain;
        this.domainConfidence = Math.max(0d, Math.min(1d, newConfidence));
        this.domainReclassified = true;
        this.assignedWorker = "";
        this.errorSummary = reason == null ? "" : reason;
        this.retryHint = "";
        this.candidateWorkers.clear();
        if (newCandidates != null) {
            this.candidateWorkers.addAll(newCandidates);
        }
        this.candidateIndex = 0;
        transitionTo(TaskStepStatus.RECLASSIFY_PENDING, this.errorSummary);
    }

    public void markSuccess(String summary) {
        this.resultSummary = summary == null ? "" : summary;
        this.errorSummary = "";
        this.retryHint = "";
        transitionTo(TaskStepStatus.SUCCESS, "Step succeeded.");
    }

    public void markFailed(String error) {
        this.errorSummary = error == null ? "" : error;
        this.retryHint = "";
        transitionTo(TaskStepStatus.FAILED, this.errorSummary);
    }

    public void markSkipped(String reason) {
        this.errorSummary = reason == null ? "" : reason;
        this.retryHint = "";
        transitionTo(TaskStepStatus.SKIPPED, this.errorSummary);
    }

    private void transitionTo(TaskStepStatus nextStatus, String reason) {
        if (nextStatus == null) {
            return;
        }
        TaskStepStatus current = this.status;
        String safeReason = reason == null ? "" : reason.trim();
        if (current != nextStatus && !canTransitionTo(nextStatus)) {
            String invalid = "Invalid transition: " + current + " -> " + nextStatus
                    + (safeReason.isEmpty() ? "" : (" (" + safeReason + ")"));
            this.errorSummary = invalid;
            this.transitionReason = invalid;
            this.transitionHistory.add(invalid);
            return;
        }
        this.status = nextStatus;
        this.transitionReason = safeReason;
        String entry = current + " -> " + nextStatus + (safeReason.isEmpty() ? "" : (": " + safeReason));
        this.transitionHistory.add(entry);
    }

    private boolean canTransitionTo(TaskStepStatus nextStatus) {
        if (nextStatus == null) {
            return false;
        }
        if (this.status == nextStatus) {
            return true;
        }
        if (isTerminal()) {
            return false;
        }
        if (this.status == TaskStepStatus.PENDING
                || this.status == TaskStepStatus.RETRY_PENDING
                || this.status == TaskStepStatus.RECLASSIFY_PENDING) {
            return nextStatus == TaskStepStatus.RUNNING || nextStatus == TaskStepStatus.SKIPPED;
        }
        if (this.status == TaskStepStatus.RUNNING) {
            return nextStatus == TaskStepStatus.SUCCESS
                    || nextStatus == TaskStepStatus.FAILED
                    || nextStatus == TaskStepStatus.SKIPPED
                    || nextStatus == TaskStepStatus.RETRY_PENDING
                    || nextStatus == TaskStepStatus.RECLASSIFY_PENDING;
        }
        return false;
    }
}
