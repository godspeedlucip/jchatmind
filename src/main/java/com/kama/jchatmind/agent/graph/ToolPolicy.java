package com.kama.jchatmind.agent.graph;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public class ToolPolicy {

    private final String policyName;
    private final Set<String> allowedTools;
    private final Set<String> requiredTools;
    private final Set<String> forbiddenTools;
    private final Set<Capability> requiredCapabilities;
    private final Set<String> requiredEvidence;
    private final Set<String> requiredSideEffects;
    private final boolean requireToolExecution;

    private ToolPolicy(Builder builder) {
        this.policyName = builder.policyName == null ? "" : builder.policyName.trim();
        this.allowedTools = normalize(builder.allowedTools);
        this.requiredTools = normalize(builder.requiredTools);
        this.forbiddenTools = normalize(builder.forbiddenTools);
        this.requiredCapabilities = builder.requiredCapabilities == null
                ? new LinkedHashSet<Capability>()
                : new LinkedHashSet<Capability>(builder.requiredCapabilities);
        this.requiredEvidence = normalize(builder.requiredEvidence);
        this.requiredSideEffects = normalize(builder.requiredSideEffects);
        this.requireToolExecution = builder.requireToolExecution;
    }

    public String getPolicyName() {
        return policyName;
    }

    public Set<String> getAllowedTools() {
        return Collections.unmodifiableSet(allowedTools);
    }

    public Set<String> getRequiredTools() {
        return Collections.unmodifiableSet(requiredTools);
    }

    public Set<String> getForbiddenTools() {
        return Collections.unmodifiableSet(forbiddenTools);
    }

    public Set<Capability> getRequiredCapabilities() {
        return Collections.unmodifiableSet(requiredCapabilities);
    }

    public Set<String> getRequiredEvidence() {
        return Collections.unmodifiableSet(requiredEvidence);
    }

    public Set<String> getRequiredSideEffects() {
        return Collections.unmodifiableSet(requiredSideEffects);
    }

    public boolean isRequireToolExecution() {
        return requireToolExecution;
    }

    public boolean isAllowedTool(String toolName) {
        if (toolName == null || toolName.trim().isEmpty()) {
            return false;
        }
        if (allowedTools.isEmpty()) {
            return true;
        }
        String normalized = toolName.trim().toLowerCase();
        for (String allowed : allowedTools) {
            if (normalized.equals(allowed)) {
                return true;
            }
        }
        return false;
    }

    public boolean isForbiddenTool(String toolName) {
        if (toolName == null || toolName.trim().isEmpty()) {
            return false;
        }
        String normalized = toolName.trim().toLowerCase();
        for (String forbidden : forbiddenTools) {
            if (normalized.equals(forbidden)) {
                return true;
            }
        }
        return false;
    }

    public List<String> allowedToolList() {
        return new ArrayList<String>(allowedTools);
    }

    private Set<String> normalize(Set<String> values) {
        Set<String> out = new LinkedHashSet<String>();
        if (values == null) {
            return out;
        }
        for (String value : values) {
            if (value != null && !value.trim().isEmpty()) {
                out.add(value.trim().toLowerCase());
            }
        }
        return out;
    }

    public static Builder builder(String policyName) {
        return new Builder(policyName);
    }

    public static class Builder {
        private final String policyName;
        private Set<String> allowedTools = new LinkedHashSet<String>();
        private Set<String> requiredTools = new LinkedHashSet<String>();
        private Set<String> forbiddenTools = new LinkedHashSet<String>();
        private Set<Capability> requiredCapabilities = new LinkedHashSet<Capability>();
        private Set<String> requiredEvidence = new LinkedHashSet<String>();
        private Set<String> requiredSideEffects = new LinkedHashSet<String>();
        private boolean requireToolExecution = false;

        private Builder(String policyName) {
            this.policyName = policyName;
        }

        public Builder allow(String... tools) {
            addAll(this.allowedTools, tools);
            return this;
        }

        public Builder requireTools(String... tools) {
            addAll(this.requiredTools, tools);
            return this;
        }

        public Builder forbid(String... tools) {
            addAll(this.forbiddenTools, tools);
            return this;
        }

        public Builder requireCapabilities(Capability... capabilities) {
            if (capabilities != null) {
                for (Capability capability : capabilities) {
                    if (capability != null) {
                        this.requiredCapabilities.add(capability);
                    }
                }
            }
            return this;
        }

        public Builder requireEvidence(String... evidenceKeys) {
            addAll(this.requiredEvidence, evidenceKeys);
            return this;
        }

        public Builder requireSideEffects(String... sideEffects) {
            addAll(this.requiredSideEffects, sideEffects);
            return this;
        }

        public Builder requireToolExecution(boolean requireToolExecution) {
            this.requireToolExecution = requireToolExecution;
            return this;
        }

        public ToolPolicy build() {
            return new ToolPolicy(this);
        }

        private void addAll(Set<String> target, String... values) {
            if (target == null || values == null) {
                return;
            }
            for (String value : values) {
                if (value != null && !value.trim().isEmpty()) {
                    target.add(value.trim());
                }
            }
        }
    }
}
