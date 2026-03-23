# Java中Redis的使用概览
Redis是一个高性能的内存键值数据库，常用于缓存、会话管理、分布式锁、消息队列与排行榜。在Java项目中，最常见的集成方式是Spring Data Redis配合Lettuce客户端。

在典型的后端系统里，Redis常作为“热数据层”，而PostgreSQL/MySQL作为“冷数据层”。访问流程通常是先查Redis，未命中再查数据库，并将结果回写Redis。

# Java集成Redis的常见方案
## Jedis
Jedis是较早期的Java Redis客户端，API直观，适合简单场景。

## Lettuce
Lettuce是基于Netty的线程安全客户端，支持同步、异步与响应式编程。Spring Boot默认使用Lettuce。

## Redisson
Redisson在Redis基础上封装了分布式对象与分布式并发工具，例如分布式锁、延迟队列、布隆过滤器等，适合业务复杂场景。

# Spring Boot中Redis基础配置
在Spring Boot中通常通过application.yaml配置Redis连接信息。常见字段包括host、port、database、username和password。

```yaml
spring:
  data:
    redis:
      host: localhost
      port: 6379
      database: 0
      username: default
      password: your_redis_password
```

如果Redis开启了认证但未配置password，常见报错是`NOAUTH Authentication required`。

# Spring Data Redis核心组件
## RedisTemplate
`RedisTemplate<K, V>`是最常用的操作入口，支持字符串、哈希、列表、集合、有序集合等结构。

## StringRedisTemplate
`StringRedisTemplate`是`RedisTemplate<String, String>`的特化版本，适合纯字符串场景。

## 序列化策略
生产环境应明确key与value序列化策略，常见组合如下：

| 维度 | 推荐配置 |
| --- | --- |
| Key序列化 | StringRedisSerializer |
| Value序列化 | GenericJackson2JsonRedisSerializer |
| HashKey序列化 | StringRedisSerializer |
| HashValue序列化 | GenericJackson2JsonRedisSerializer |

# Java中Redis常用数据结构实践
## String类型
适合缓存对象JSON、计数器、短期令牌。

典型操作：`set`、`get`、`incr`、`decr`。

## Hash类型
适合结构化对象字段存储，如用户资料、会话属性。

典型操作：`hset`、`hget`、`hgetall`。

## List类型
适合消息流、任务队列、时间序列事件。

典型操作：`lpush`、`rpop`、`llen`。

## Set类型
适合去重集合、标签集合、关系判定。

典型操作：`sadd`、`smembers`、`sismember`。

## ZSet类型
适合排行榜、带权重的推荐排序。

典型操作：`zadd`、`zrevrange`、`zscore`。

# 缓存设计与失效策略
## Cache Aside模式
应用先读缓存，未命中再读数据库并回写缓存。更新时先更新数据库，再删除缓存。

## 过期时间策略
为避免雪崩，TTL建议加入随机抖动。例如基础TTL为30分钟，可再增加0-5分钟随机值。

## 穿透与击穿
- 穿透：查询不存在数据。可缓存空值或使用布隆过滤器。
- 击穿：热点Key在过期瞬间大量请求击中数据库。可用互斥锁或逻辑过期。

# Redis在Java项目中的高级用法
## 分布式锁
可通过`SET key value NX PX`实现基础分布式锁，释放锁时要校验value一致性。

## 发布订阅
使用`publish/subscribe`实现轻量消息通知，但不保证离线消息持久化。

## 延迟队列
可用`ZSet`的score存时间戳，轮询到期任务执行。

# 连接池与性能调优建议
- 控制连接池大小，避免过大导致资源浪费。
- 设置合理超时（连接超时、命令超时）。
- 对大Key进行治理，避免单次操作阻塞。
- 尽量使用批量操作与Pipeline提升吞吐。

# Java与Redis常见问题排查
## NOAUTH Authentication required
说明Redis开启认证但客户端未提供用户名或密码。

## Connection refused
说明Redis服务未启动、端口错误或网络不可达。

## 反序列化异常
说明写入与读取使用的序列化器不一致，或DTO缺少可反序列化构造器。

## 热点Key导致延迟抖动
应增加本地缓存、多副本读、请求合并与过期错峰策略。

# 示例：会话消息缓存场景
在聊天系统中，可将最近N条消息缓存到Redis List：
1. 新消息写入数据库。
2. 同步更新Redis会话窗口。
3. 读取会话时优先读取Redis，失败或未命中再回源数据库。
4. Redis不可用时降级到数据库，保证主流程可用。

# 结语
Java集成Redis的关键不是“能连上”，而是“在故障和高并发下仍可用”。建议优先做好认证配置、序列化一致性、缓存降级策略与监控告警。
