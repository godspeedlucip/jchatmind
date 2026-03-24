package com.kama.jchatmind.agent.graph;

public enum TaskStepStatus {
    PENDING,
    RETRY_PENDING,
    RECLASSIFY_PENDING,
    RUNNING,
    SUCCESS,
    FAILED,
    SKIPPED
}
