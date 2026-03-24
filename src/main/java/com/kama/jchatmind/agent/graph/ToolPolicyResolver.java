package com.kama.jchatmind.agent.graph;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Set;

public class ToolPolicyResolver {

    public ToolPolicy resolve(TaskStep step) {
        if (step == null) {
            return ToolPolicy.builder("GENERAL")
                    .requireToolExecution(false)
                    .build();
        }
        if (step.getDomain() == TaskDomain.SQL) {
            return ToolPolicy.builder("SQL_READ")
                    .allow("databasequery")
                    .requireTools("databasequery")
                    .forbid("sendemail", "emailtool", "weathertool", "weather", "knowledgetool", "websearchtool")
                    .requireCapabilities(Capability.DATABASE_READ)
                    .requireToolExecution(true)
                    .build();
        }
        if (step.getDomain() == TaskDomain.RAG) {
            return ToolPolicy.builder("RAG_RETRIEVE")
                    .allow("knowledgetool")
                    .requireTools("knowledgetool")
                    .forbid("databasequery", "sendemail", "emailtool")
                    .requireCapabilities(Capability.KNOWLEDGE_RETRIEVE)
                    .requireToolExecution(true)
                    .build();
        }
        if (step.getDomain() == TaskDomain.TOOL) {
            String desc = step.getDescription() == null ? "" : step.getDescription().toLowerCase(Locale.ROOT);
            if (isEmailStep(desc)) {
                return ToolPolicy.builder("EMAIL_DELIVERY")
                        .allow("sendemail")
                        .requireTools("sendemail")
                        .forbid("databasequery", "knowledgetool")
                        .requireCapabilities(Capability.EMAIL_SEND)
                        .requireSideEffects("email_sent")
                        .requireToolExecution(true)
                        .build();
            }
            if (isWeatherStep(desc)) {
                return ToolPolicy.builder("WEATHER_QUERY")
                        .allow("weather", "getweather")
                        .requireTools("weather")
                        .forbid("getdate", "datetool", "databasequery", "sendemail")
                        .requireCapabilities(Capability.WEATHER_READ)
                        .requireEvidence("weather_result")
                        .requireToolExecution(true)
                        .build();
            }
            return ToolPolicy.builder("TOOL_GENERIC")
                    .requireToolExecution(true)
                    .build();
        }
        return ToolPolicy.builder("GENERAL")
                .requireToolExecution(false)
                .build();
    }

    public Set<Capability> capabilitiesForTool(String toolName) {
        if (toolName == null || toolName.trim().isEmpty()) {
            return Collections.emptySet();
        }
        String normalized = toolName.trim().toLowerCase(Locale.ROOT);
        Set<Capability> capabilities = new LinkedHashSet<Capability>();
        if ("databasequery".equals(normalized) || "databasetool".equals(normalized)) {
            capabilities.add(Capability.DATABASE_READ);
        }
        if ("knowledgetool".equals(normalized)) {
            capabilities.add(Capability.KNOWLEDGE_RETRIEVE);
        }
        if ("weather".equals(normalized) || "getweather".equals(normalized)) {
            capabilities.add(Capability.WEATHER_READ);
        }
        if ("sendemail".equals(normalized) || "emailtool".equals(normalized)) {
            capabilities.add(Capability.EMAIL_SEND);
        }
        return capabilities;
    }

    private boolean isEmailStep(String descLower) {
        return descLower.contains("email")
                || descLower.contains("mail")
                || descLower.contains("\u90ae\u4ef6")
                || descLower.contains("\u90ae\u7bb1")
                || descLower.contains("\u53d1\u9001");
    }

    private boolean isWeatherStep(String descLower) {
        return descLower.contains("weather")
                || descLower.contains("forecast")
                || descLower.contains("\u5929\u6c14")
                || descLower.contains("\u6c14\u6e29");
    }
}
