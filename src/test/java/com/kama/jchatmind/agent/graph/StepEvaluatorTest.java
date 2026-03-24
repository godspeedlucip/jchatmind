package com.kama.jchatmind.agent.graph;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StepEvaluatorTest {

    private final ToolPolicyResolver resolver = new ToolPolicyResolver();
    private final StepEvaluator evaluator = new StepEvaluator(resolver);

    @Test
    void sqlStepWrongToolShouldRetryAsPolicyViolation() {
        TaskStep step = new TaskStep("s1", "查询数据库的agent表内容", TaskDomain.SQL, 0.95d, false, List.of("SQL_WORKER"));
        StepEvidence evidence = new StepEvidence();
        evidence.setToolCalls(List.of("emailTool"));
        evidence.setError("Tool execution blocked: worker SQL_WORKER cannot call tool 'emailTool'.");

        StepEvaluationResult result = evaluator.evaluate(step, evidence, resolver.resolve(step), true, evidence.getError());

        assertEquals(StepDecision.RETRY_SAME_WORKER, result.getDecision());
        assertTrue(result.isPolicyViolation());
        assertTrue(evidence.isPolicyViolation());
    }

    @Test
    void weatherDateOnlyShouldRetry() {
        TaskStep step = new TaskStep("s2", "查看深圳的天气", TaskDomain.TOOL, 0.92d, false, List.of("TOOL_WORKER"));
        StepEvidence evidence = new StepEvidence();
        evidence.setToolCalls(List.of("getDate"));
        evidence.setToolResults(List.of("getDate -> 2026-03-24"));
        evidence.setToolExecutionAttempted(true);
        evidence.setToolExecutionSucceeded(true);

        StepEvaluationResult result = evaluator.evaluate(step, evidence, resolver.resolve(step), false, "");

        assertEquals(StepDecision.RETRY_SAME_WORKER, result.getDecision());
        assertTrue(result.isPolicyViolation());
        assertEquals("required_tool_missing", evidence.getPolicyCheckResult());
    }

    @Test
    void emailStepWithoutSendEmailShouldRetry() {
        TaskStep step = new TaskStep("s3", "发送邮件给xxx", TaskDomain.TOOL, 0.90d, true, List.of("TOOL_WORKER"));
        StepEvidence evidence = new StepEvidence();
        evidence.setTextAnswer("我已经帮你整理好邮件内容。");

        StepEvaluationResult result = evaluator.evaluate(step, evidence, resolver.resolve(step), false, "");

        assertEquals(StepDecision.RETRY_SAME_WORKER, result.getDecision());
        assertTrue(result.isPolicyViolation());
    }

    @Test
    void emailStepWithSideEffectShouldPass() {
        TaskStep step = new TaskStep("s4", "发送邮件给xxx", TaskDomain.TOOL, 0.90d, true, List.of("TOOL_WORKER"));
        StepEvidence evidence = new StepEvidence();
        evidence.setToolCalls(List.of("sendEmail"));
        evidence.setToolResults(List.of("sendEmail -> 邮件已提交发送！收件人: a@b.com"));
        evidence.setToolExecutionAttempted(true);
        evidence.setToolExecutionSucceeded(true);

        StepEvaluationResult result = evaluator.evaluate(step, evidence, resolver.resolve(step), false, "");

        assertEquals(StepDecision.PASS, result.getDecision());
        assertTrue(evidence.isSideEffectObserved());
    }

    @Test
    void domainMismatchShouldReclassify() {
        TaskStep step = new TaskStep("s5", "介绍redis集成", TaskDomain.RAG, 0.80d, false, List.of("RAG_WORKER"));
        StepEvidence evidence = new StepEvidence();
        evidence.setError("Worker SQL_WORKER is not compatible with domain RAG. domain mismatch");

        StepEvaluationResult result = evaluator.evaluate(step, evidence, resolver.resolve(step), true, evidence.getError());

        assertEquals(StepDecision.RECLASSIFY, result.getDecision());
    }

    @Test
    void sqlExecutionErrorShouldRetry() {
        TaskStep step = new TaskStep("s6", "查询不存在的表", TaskDomain.SQL, 0.95d, false, List.of("SQL_WORKER"));
        StepEvidence evidence = new StepEvidence();
        evidence.setToolCalls(List.of("databaseQuery"));
        evidence.setToolResults(List.of("databaseQuery -> SQL 执行失败：表 'not_exists' 不存在"));
        evidence.setToolExecutionAttempted(true);
        evidence.setToolExecutionSucceeded(true);

        StepEvaluationResult result = evaluator.evaluate(step, evidence, resolver.resolve(step), false, "");

        assertEquals(StepDecision.RETRY_SAME_WORKER, result.getDecision());
        assertEquals("sql_execution_failed", evidence.getPolicyCheckResult());
        assertTrue(!result.isPolicyViolation());
    }
}
