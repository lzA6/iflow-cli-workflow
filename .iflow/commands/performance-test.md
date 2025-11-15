---
name: performance-test
description: "系统性能基准测试。当系统需要进行：(1) 性能基准测试，(2) 响应时间评估，(3) 并发能力测试，(4) 资源使用分析，或任何性能相关测试任务时使用"
license: Proprietary. LICENSE.txt has complete terms
workflow_trigger: /performance-test
agent_path: .iflow/agents/performance-analyzer
---

# 系统性能基准测试工作流

## 功能概述

这个工作流执行全面的系统性能基准测试，包括响应时间、并发能力、资源使用等关键指标的评估。

## 执行步骤

### 1. 性能环境初始化
- 激活超级思考模式
- 初始化性能监控环境
- 加载基准测试配置

### 2. 全面性能测试
- **响应时间测试**: 测试P95、P99响应时间
- **并发能力测试**: 10-50线程并发访问测试
- **资源使用分析**: CPU、内存、磁盘使用率监控
- **吞吐量评估**: QPS/TPS性能指标测试

### 3. 性能优化建议
- 识别性能瓶颈
- 提供优化策略
- 生成性能报告
- 制定改进计划

## 输入参数

```
/performance-test [可选参数]
```

**可选参数**:
- `--benchmark`: 综合基准测试
- `--stress`: 压力测试模式
- `--memory`: 内存使用专项测试
- `--concurrent`: 并发能力测试
- `--detailed`: 详细性能分析

## 输出结果

### 性能测试报告
```json
{
  "test_summary": {
    "overall_performance": "excellent/good/fair/poor",
    "p95_response_time": 1.2,
    "p99_response_time": 3.2,
    "concurrent_capacity": 50,
    "throughput_qps": 1000
  },
  "detailed_metrics": {
    "response_time_analysis": {...},
    "concurrency_test": {...},
    "resource_usage": {...},
    "throughput_analysis": {...}
  },
  "performance_recommendations": [...],
  "optimization_plan": [...]
}
```

## 使用示例

### 基础性能测试
```
/performance-test
```

### 综合基准测试
```
/performance-test --benchmark
```

### 压力测试
```
/performance-test --stress
```

### 内存专项测试
```
/performance-test --memory
```

## 技术要求

- 需要访问性能监控核心模块
- 支持多线程并发测试
- 需要资源使用监控权限
- 支持实时性能指标收集

## 测试项目

### 响应时间测试
- P50响应时间
- P95响应时间
- P99响应时间
- 最大响应时间

### 并发能力测试
- 10线程并发测试
- 25线程并发测试
- 50线程并发测试
- 并发稳定性评估

### 资源使用监控
- CPU使用率监控
- 内存使用率监控
- 磁盘I/O监控
- 网络带宽监控

### 吞吐量测试
- QPS (Queries Per Second)
- TPS (Transactions Per Second)
- 请求处理能力
- 系统承载能力

## 性能标准

### 优秀 (Excellent)
- P99响应时间 < 2秒
- 并发能力 ≥ 50线程
- 资源使用率 < 70%
- 吞吐量 ≥ 1000 QPS

### 良好 (Good)
- P99响应时间 < 5秒
- 并发能力 ≥ 25线程
- 资源使用率 < 80%
- 吞吐量 ≥ 500 QPS

### 一般 (Fair)
- P99响应时间 < 10秒
- 并发能力 ≥ 10线程
- 资源使用率 < 90%
- 吞吐量 ≥ 100 QPS

### 需改进 (Poor)
- P99响应时间 ≥ 10秒
- 并发能力 < 10线程
- 资源使用率 ≥ 90%
- 吞吐量 < 100 QPS

## 注意事项

1. 基准测试可能需要5-10分钟
2. 压力测试会影响系统性能，建议在测试环境进行
3. 内存测试可能暂时增加内存使用
4. 并发测试会模拟真实负载，请确保系统稳定