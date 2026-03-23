package com.kama.jchatmind.agent.tools;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.jdbc.BadSqlGrammarException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Component
@Slf4j
public class DataBaseTools implements com.kama.jchatmind.agent.tools.Tool {

    private static final Pattern MISSING_RELATION_PATTERN =
            Pattern.compile("relation\\s+\"([^\"]+)\"\\s+does\\s+not\\s+exist", Pattern.CASE_INSENSITIVE);

    private static final String LIST_TABLE_SQL =
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name";

    private final JdbcTemplate jdbcTemplate;

    public DataBaseTools(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    @Override
    public String getName() {
        return "dataBaseTool";
    }

    @Override
    public String getDescription() {
        return "用于在 PostgreSQL 中执行只读查询（SELECT），并返回结构化结果。";
    }

    @Override
    public ToolType getType() {
        return ToolType.OPTIONAL;
    }

    @Tool(
            name = "databaseQuery",
            description = "在 PostgreSQL 中执行只读查询（仅 SELECT）。禁止 INSERT/UPDATE/DELETE/DDL。"
    )
    public String query(String sql) {
        try {
            String trimmed = sql == null ? "" : sql.trim();
            String upper = trimmed.toUpperCase(Locale.ROOT);
            if (!upper.startsWith("SELECT")) {
                log.warn("Rejected non-SELECT SQL: {}", sql);
                return "仅允许执行 SELECT 查询。\nSQL: " + sql;
            }

            List<String> rows = jdbcTemplate.query(trimmed, (ResultSet rs) -> {
                List<String> resultRows = new ArrayList<>();
                ResultSetMetaData metaData = rs.getMetaData();
                int columnCount = metaData.getColumnCount();

                if (columnCount == 0) {
                    return List.of("查询成功，但没有可显示的列。");
                }

                List<String> columnNames = new ArrayList<>();
                List<Integer> columnWidths = new ArrayList<>();
                for (int i = 1; i <= columnCount; i++) {
                    String columnName = metaData.getColumnName(i);
                    columnNames.add(columnName);
                    columnWidths.add(columnName.length());
                }

                List<List<String>> dataRows = new ArrayList<>();
                while (rs.next()) {
                    List<String> rowData = new ArrayList<>();
                    for (int i = 1; i <= columnCount; i++) {
                        Object value = rs.getObject(i);
                        String valueStr = value == null ? "NULL" : value.toString();
                        rowData.add(valueStr);
                        if (valueStr.length() > columnWidths.get(i - 1)) {
                            columnWidths.set(i - 1, valueStr.length());
                        }
                    }
                    dataRows.add(rowData);
                }

                StringBuilder header = new StringBuilder("| ");
                for (int i = 0; i < columnCount; i++) {
                    header.append(String.format("%-" + columnWidths.get(i) + "s", columnNames.get(i))).append(" | ");
                }
                resultRows.add(header.toString());

                StringBuilder separator = new StringBuilder("|");
                for (int i = 0; i < columnCount; i++) {
                    separator.append("-".repeat(columnWidths.get(i) + 2)).append("|");
                }
                resultRows.add(separator.toString());

                if (dataRows.isEmpty()) {
                    int totalWidth = columnWidths.stream().mapToInt(w -> w + 3).sum() - 1;
                    resultRows.add("| " + String.format("%-" + Math.max(totalWidth - 2, 1) + "s", "(无数据)") + " |");
                } else {
                    for (List<String> rowData : dataRows) {
                        StringBuilder row = new StringBuilder("| ");
                        for (int i = 0; i < columnCount; i++) {
                            row.append(String.format("%-" + columnWidths.get(i) + "s", rowData.get(i))).append(" | ");
                        }
                        resultRows.add(row.toString());
                    }
                }

                return resultRows;
            });

            return "查询结果:\n" + String.join("\n", rows);
        } catch (BadSqlGrammarException e) {
            String rootMessage = extractRootMessage(e);
            String lower = rootMessage.toLowerCase(Locale.ROOT);
            if (lower.contains("relation") && lower.contains("does not exist")) {
                String missingTable = extractMissingRelationName(rootMessage);
                List<String> tables = listPublicTables();
                String tableList = tables.isEmpty() ? "(未获取到表清单)" : String.join(", ", tables);
                return "SQL 执行失败：表 "
                        + (missingTable.isEmpty() ? "(未知)" : ("'" + missingTable + "'"))
                        + " 不存在。\n"
                        + "请改用已存在的表后重试。\n"
                        + "当前 public schema 表：" + tableList + "\n"
                        + "建议先执行：" + LIST_TABLE_SQL;
            }
            log.error("SQL grammar error: {}", rootMessage, e);
            return "SQL 语法错误：" + rootMessage + "\nSQL: " + sql;
        } catch (Exception e) {
            log.error("Database query failed", e);
            return "数据库查询失败：" + extractRootMessage(e) + "\nSQL: " + sql;
        }
    }

    private String extractRootMessage(Throwable throwable) {
        Throwable cursor = throwable;
        while (cursor.getCause() != null) {
            cursor = cursor.getCause();
        }
        if (cursor.getMessage() != null && !cursor.getMessage().isBlank()) {
            return cursor.getMessage();
        }
        return throwable.getMessage() == null ? "未知错误" : throwable.getMessage();
    }

    private String extractMissingRelationName(String message) {
        Matcher matcher = MISSING_RELATION_PATTERN.matcher(message);
        return matcher.find() ? matcher.group(1) : "";
    }

    private List<String> listPublicTables() {
        try {
            return jdbcTemplate.queryForList(LIST_TABLE_SQL, String.class)
                    .stream()
                    .limit(100)
                    .toList();
        } catch (Exception e) {
            log.warn("Failed to list tables from information_schema: {}", e.getMessage());
            return List.of();
        }
    }
}
