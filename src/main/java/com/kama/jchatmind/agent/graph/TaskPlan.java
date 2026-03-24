package com.kama.jchatmind.agent.graph;

import java.util.ArrayList;
import java.util.List;

public class TaskPlan {

    private final String originalRequest;
    private final List<TaskStep> steps;
    private TaskPlanStatus status;

    public TaskPlan(String originalRequest, List<TaskStep> steps) {
        this.originalRequest = originalRequest == null ? "" : originalRequest;
        this.steps = steps == null ? new ArrayList<TaskStep>() : steps;
        this.status = TaskPlanStatus.PENDING;
    }

    public String getOriginalRequest() {
        return originalRequest;
    }

    public List<TaskStep> getSteps() {
        return steps;
    }

    public TaskPlanStatus getStatus() {
        return status;
    }

    public void markRunning() {
        this.status = TaskPlanStatus.RUNNING;
    }

    public void updateFinalStatus() {
        boolean hasFailure = false;
        for (TaskStep step : steps) {
            if (step.getStatus() == TaskStepStatus.FAILED) {
                hasFailure = true;
                break;
            }
        }
        this.status = hasFailure ? TaskPlanStatus.COMPLETED_WITH_FAILURE : TaskPlanStatus.COMPLETED;
    }

    public TaskStep findById(String id) {
        if (id == null || id.trim().isEmpty()) {
            return null;
        }
        for (TaskStep step : steps) {
            if (id.equals(step.getId())) {
                return step;
            }
        }
        return null;
    }

    public TaskStep nextRunnableStep() {
        // First pass: non-delivery tasks.
        for (TaskStep step : steps) {
            if (!step.isDeliveryStage() && step.isRunnable()) {
                return step;
            }
        }
        // Second pass: delivery tasks only after all non-delivery tasks are terminal.
        if (!allNonDeliveryTerminal()) {
            return null;
        }
        for (TaskStep step : steps) {
            if (step.isDeliveryStage() && step.isRunnable()) {
                return step;
            }
        }
        return null;
    }

    public boolean allStepsTerminal() {
        for (TaskStep step : steps) {
            if (!step.isTerminal()) {
                return false;
            }
        }
        return true;
    }

    private boolean allNonDeliveryTerminal() {
        for (TaskStep step : steps) {
            if (!step.isDeliveryStage() && !step.isTerminal()) {
                return false;
            }
        }
        return true;
    }
}
