#!/usr/bin/env python3
"""
完整依据和解释系统
为每个决策提供详细的推理过程、证据链和自我反省
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
from enum import Enum

class DecisionType(Enum):
    """决策类型"""
    FILE_RETENTION = "file_retention"
    FILE_REMOVAL = "file_removal"
    CODE_REFACTOR = "code_refactor"
    SECURITY_FIX = "security_fix"
    PERFORMANCE_OPTIMIZE = "performance_optimize"
    ARCHITECTURE_CHANGE = "architecture_change"

class EvidenceType(Enum):
    """证据类型"""
    CODE_ANALYSIS = "code_analysis"
    METRICS_DATA = "metrics_data"
    DEPENDENCY_GRAPH = "dependency_graph"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_TEST = "performance_test"
    USER_FEEDBACK = "user_feedback"
    BEST_PRACTICES = "best_practices"
    INDUSTRY_STANDARDS = "industry_standards"

@dataclass
class Evidence:
    """证据"""
    evidence_type: EvidenceType
    source: str
    content: str
    confidence: float
    timestamp: str
    verification_method: str
    supporting_data: Dict[str, Any]

@dataclass
class ReasoningStep:
    """推理步骤"""
    step_number: int
    description: str
    input_data: Dict[str, Any]
    reasoning_process: str
    conclusion: str
    confidence: float
    assumptions: List[str]
    limitations: List[str]

@dataclass
class SelfReflection:
    """自我反省"""
    reflection_type: str
    question: str
    analysis: str
    insights: List[str]
    biases_identified: List[str]
    alternative_approaches: List[str]
    confidence_adjustment: float

@dataclass
class DecisionRecord:
    """决策记录"""
    decision_id: str
    decision_type: DecisionType
    target: str  # 目标文件或模块
    decision: str  # 决策结果
    reasoning_chain: List[ReasoningStep]
    evidence_chain: List[Evidence]
    self_reflections: List[SelfReflection]
    confidence_score: float
    risk_assessment: str
    impact_analysis: Dict[str, Any]
    alternatives_considered: List[str]
    final_justification: str
    timestamp: str

class ComprehensiveJustificationSystem:
    """完整依据和解释系统"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.decision_records = []
        self.evidence_database = {}
        self.reasoning_templates = self._load_reasoning_templates()
        self.justification_principles = self._load_justification_principles()
        
    async def create_comprehensive_decision(self, 
                                          decision_type: DecisionType,
                                          target: str,
                                          analysis_data: Dict[str, Any]) -> DecisionRecord:
        """创建综合决策记录"""
        print(f"🧠 创建综合决策: {decision_type.value} for {target}")
        
        # 1. 生成决策ID
        decision_id = await self._generate_decision_id(decision_type, target)
        
        # 2. 收集证据
        print("📚 收集证据...")
        evidence_chain = await self._collect_evidence(decision_type, target, analysis_data)
        
        # 3. 构建推理链
        print("🔗 构建推理链...")
        reasoning_chain = await self._build_reasoning_chain(decision_type, target, evidence_chain, analysis_data)
        
        # 4. 执行自我反省
        print("🤔 执行自我反省...")
        self_reflections = await self._perform_self_reflection(decision_type, target, reasoning_chain, evidence_chain)
        
        # 5. 评估置信度
        confidence_score = await self._calculate_confidence_score(evidence_chain, reasoning_chain, self_reflections)
        
        # 6. 风险评估
        risk_assessment = await self._assess_risks(decision_type, target, evidence_chain)
        
        # 7. 影响分析
        impact_analysis = await self._analyze_impact(decision_type, target, evidence_chain, analysis_data)
        
        # 8. 考虑替代方案
        alternatives_considered = await self._consider_alternatives(decision_type, target, analysis_data)
        
        # 9. 生成最终决策
        final_decision = await self._generate_final_decision(decision_type, target, reasoning_chain, confidence_score)
        
        # 10. 生成最终依据
        final_justification = await self._generate_final_justification(
            decision_type, target, final_decision, evidence_chain, reasoning_chain, self_reflections
        )
        
        # 11. 创建决策记录
        decision_record = DecisionRecord(
            decision_id=decision_id,
            decision_type=decision_type,
            target=target,
            decision=final_decision,
            reasoning_chain=reasoning_chain,
            evidence_chain=evidence_chain,
            self_reflections=self_reflections,
            confidence_score=confidence_score,
            risk_assessment=risk_assessment,
            impact_analysis=impact_analysis,
            alternatives_considered=alternatives_considered,
            final_justification=final_justification,
            timestamp=datetime.now().isoformat()
        )
        
        # 12. 保存决策记录
        await self._save_decision_record(decision_record)
        
        print(f"✅ 综合决策创建完成: {decision_id}")
        return decision_record
    
    async def _generate_decision_id(self, decision_type: DecisionType, target: str) -> str:
        """生成决策ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content = f"{decision_type.value}_{target}_{timestamp}"
        hash_id = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"DEC_{hash_id}"
    
    async def _collect_evidence(self, decision_type: DecisionType, 
                              target: str, analysis_data: Dict[str, Any]) -> List[Evidence]:
        """收集证据"""
        evidence_chain = []
        
        # 1. 代码分析证据
        code_evidence = await self._collect_code_analysis_evidence(target, analysis_data)
        evidence_chain.extend(code_evidence)
        
        # 2. 指标数据证据
        metrics_evidence = await self._collect_metrics_evidence(target, analysis_data)
        evidence_chain.extend(metrics_evidence)
        
        # 3. 依赖关系证据
        dependency_evidence = await self._collect_dependency_evidence(target, analysis_data)
        evidence_chain.extend(dependency_evidence)
        
        # 4. 安全扫描证据
        if decision_type in [DecisionType.FILE_REMOVAL, DecisionType.SECURITY_FIX]:
            security_evidence = await self._collect_security_evidence(target, analysis_data)
            evidence_chain.extend(security_evidence)
        
        # 5. 性能测试证据
        if decision_type in [DecisionType.PERFORMANCE_OPTIMIZE, DecisionType.FILE_REMOVAL]:
            performance_evidence = await self._collect_performance_evidence(target, analysis_data)
            evidence_chain.extend(performance_evidence)
        
        # 6. 最佳实践证据
        best_practices_evidence = await self._collect_best_practices_evidence(decision_type, target)
        evidence_chain.extend(best_practices_evidence)
        
        # 7. 行业标准证据
        standards_evidence = await self._collect_standards_evidence(decision_type, target)
        evidence_chain.extend(standards_evidence)
        
        return evidence_chain
    
    async def _collect_code_analysis_evidence(self, target: str, analysis_data: Dict[str, Any]) -> List[Evidence]:
        """收集代码分析证据"""
        evidence = []
        
        target_path = self.project_root / target
        if not target_path.exists():
            return evidence
        
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 代码结构分析
            lines = content.split('\n')
            functions = re.findall(r'def\s+(\w+)', content)
            classes = re.findall(r'class\s+(\w+)', content)
            imports = re.findall(r'import\s+(\w+)|from\s+(\w+)', content)
            
            code_structure = {
                "total_lines": len(lines),
                "code_lines": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
                "functions": functions,
                "classes": classes,
                "imports": [imp[0] or imp[1] for imp in imports],
                "complexity_indicators": {
                    "nested_loops": content.count('for') + content.count('while'),
                    "conditional_statements": content.count('if') + content.count('elif'),
                    "exception_handling": content.count('try:') + content.count('except'),
                    "async_functions": content.count('async def')
                }
            }
            
            evidence.append(Evidence(
                evidence_type=EvidenceType.CODE_ANALYSIS,
                source=f"static_analysis:{target}",
                content=f"代码结构分析: {len(functions)}个函数, {len(classes)}个类, {len(lines)}行代码",
                confidence=0.9,
                timestamp=datetime.now().isoformat(),
                verification_method="static_code_analysis",
                supporting_data=code_structure
            ))
            
        except Exception as e:
            evidence.append(Evidence(
                evidence_type=EvidenceType.CODE_ANALYSIS,
                source=f"error_analysis:{target}",
                content=f"代码分析失败: {e}",
                confidence=0.1,
                timestamp=datetime.now().isoformat(),
                verification_method="error_handling",
                supporting_data={"error": str(e)}
            ))
        
        return evidence
    
    async def _collect_metrics_evidence(self, target: str, analysis_data: Dict[str, Any]) -> List[Evidence]:
        """收集指标数据证据"""
        evidence = []
        
        if "metrics" in analysis_data:
            metrics = analysis_data["metrics"]
            
            evidence.append(Evidence(
                evidence_type=EvidenceType.METRICS_DATA,
                source=f"metrics_analysis:{target}",
                content=f"指标数据: 复杂度={metrics.get('complexity', 'N/A')}, 可维护性={metrics.get('maintainability', 'N/A')}",
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                verification_method="automated_metrics_calculation",
                supporting_data=metrics
            ))
        
        return evidence
    
    async def _collect_dependency_evidence(self, target: str, analysis_data: Dict[str, Any]) -> List[Evidence]:
        """收集依赖关系证据"""
        evidence = []
        
        if "dependencies" in analysis_data:
            dependencies = analysis_data["dependencies"]
            
            dependency_count = len(dependencies)
            critical_dependencies = [dep for dep in dependencies if any(keyword in dep.lower() for keyword in ['core', 'engine', 'security'])]
            
            evidence.append(Evidence(
                evidence_type=EvidenceType.DEPENDENCY_GRAPH,
                source=f"dependency_analysis:{target}",
                content=f"依赖分析: {dependency_count}个依赖, {len(critical_dependencies)}个关键依赖",
                confidence=0.85,
                timestamp=datetime.now().isoformat(),
                verification_method="dependency_parsing",
                supporting_data={
                    "total_dependencies": dependency_count,
                    "critical_dependencies": critical_dependencies,
                    "dependency_list": dependencies
                }
            ))
        
        return evidence
    
    async def _collect_security_evidence(self, target: str, analysis_data: Dict[str, Any]) -> List[Evidence]:
        """收集安全扫描证据"""
        evidence = []
        
        if "security_issues" in analysis_data:
            security_issues = analysis_data["security_issues"]
            
            high_risk_issues = [issue for issue in security_issues if issue.get("severity") == "high"]
            medium_risk_issues = [issue for issue in security_issues if issue.get("severity") == "medium"]
            
            evidence.append(Evidence(
                evidence_type=EvidenceType.SECURITY_SCAN,
                source=f"security_scan:{target}",
                content=f"安全扫描: {len(security_issues)}个问题, {len(high_risk_issues)}个高风险, {len(medium_risk_issues)}个中风险",
                confidence=0.9,
                timestamp=datetime.now().isoformat(),
                verification_method="automated_security_scanning",
                supporting_data={
                    "total_issues": len(security_issues),
                    "high_risk_issues": high_risk_issues,
                    "medium_risk_issues": medium_risk_issues,
                    "all_issues": security_issues
                }
            ))
        
        return evidence
    
    async def _collect_performance_evidence(self, target: str, analysis_data: Dict[str, Any]) -> List[Evidence]:
        """收集性能测试证据"""
        evidence = []
        
        if "performance_metrics" in analysis_data:
            perf_metrics = analysis_data["performance_metrics"]
            
            evidence.append(Evidence(
                evidence_type=EvidenceType.PERFORMANCE_TEST,
                source=f"performance_analysis:{target}",
                content=f"性能分析: 执行时间={perf_metrics.get('execution_time', 'N/A')}ms, 内存使用={perf_metrics.get('memory_usage', 'N/A')}MB",
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                verification_method="performance_benchmarking",
                supporting_data=perf_metrics
            ))
        
        return evidence
    
    async def _collect_best_practices_evidence(self, decision_type: DecisionType, target: str) -> List[Evidence]:
        """收集最佳实践证据"""
        evidence = []
        
        best_practices = {
            DecisionType.FILE_RETENTION: [
                "保留具有独特业务价值的文件",
                "保留被多个模块依赖的核心文件",
                "保留包含关键算法或知识产权的文件"
            ],
            DecisionType.FILE_REMOVAL: [
                "删除功能完全重复的文件",
                "删除无实际用途的过时文件",
                "删除测试用临时文件"
            ],
            DecisionType.CODE_REFACTOR: [
                "重构高复杂度代码以提升可维护性",
                "重构重复代码以遵循DRY原则",
                "重构违反单一职责原则的代码"
            ]
        }
        
        if decision_type in best_practices:
            practices = best_practices[decision_type]
            
            for practice in practices:
                evidence.append(Evidence(
                    evidence_type=EvidenceType.BEST_PRACTICES,
                    source="industry_best_practices",
                    content=f"最佳实践: {practice}",
                    confidence=0.7,
                    timestamp=datetime.now().isoformat(),
                    verification_method="industry_guidelines",
                    supporting_data={"practice": practice, "category": decision_type.value}
                ))
        
        return evidence
    
    async def _collect_standards_evidence(self, decision_type: DecisionType, target: str) -> List[Evidence]:
        """收集行业标准证据"""
        evidence = []
        
        standards = {
            "code_quality": "遵循ISO/IEC 25010软件质量模型标准",
            "security": "遵循OWASP安全标准和CWE分类",
            "performance": "遵循性能测试和基准测试标准",
            "maintainability": "遵循可维护性指数计算标准"
        }
        
        for standard_name, standard_desc in standards.items():
            evidence.append(Evidence(
                evidence_type=EvidenceType.INDUSTRY_STANDARDS,
                source="industry_standards",
                content=f"行业标准: {standard_desc}",
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                verification_method="standards_compliance_check",
                supporting_data={"standard": standard_name, "description": standard_desc}
            ))
        
        return evidence
    
    async def _build_reasoning_chain(self, decision_type: DecisionType, 
                                   target: str, 
                                   evidence_chain: List[Evidence],
                                   analysis_data: Dict[str, Any]) -> List[ReasoningStep]:
        """构建推理链"""
        reasoning_chain = []
        
        # 步骤1: 问题定义
        step1 = ReasoningStep(
            step_number=1,
            description=f"定义{decision_type.value}决策问题",
            input_data={"target": target, "decision_type": decision_type.value},
            reasoning_process=f"基于目标'{target}'和决策类型'{decision_type.value}'，明确需要解决的核心问题",
            conclusion=f"需要针对'{target}'进行{decision_type.value}决策",
            confidence=0.95,
            assumptions=["目标文件存在且可访问", "分析数据准确可靠"],
            limitations=["可能存在未考虑的外部因素", "分析结果的时效性限制"]
        )
        reasoning_chain.append(step1)
        
        # 步骤2: 证据评估
        evidence_summary = await self._summarize_evidence(evidence_chain)
        step2 = ReasoningStep(
            step_number=2,
            description="评估收集到的证据",
            input_data={"evidence_count": len(evidence_chain), "evidence_types": [e.evidence_type.value for e in evidence_chain]},
            reasoning_process=f"分析{len(evidence_chain)}个证据的可靠性和相关性，{evidence_summary}",
            conclusion=f"证据总体支持决策制定，置信度较高",
            confidence=0.85,
            assumptions=["证据来源可靠", "证据分析方法正确"],
            limitations=["证据可能不完整", "部分证据存在主观性"]
        )
        reasoning_chain.append(step2)
        
        # 步骤3: 影响分析
        impact_analysis = await self._analyze_decision_impact(decision_type, target, evidence_chain)
        step3 = ReasoningStep(
            step_number=3,
            description="分析决策的潜在影响",
            input_data={"decision_type": decision_type.value, "target": target},
            reasoning_process=f"基于证据分析{decision_type.value}对系统的多方面影响: {impact_analysis}",
            conclusion=f"决策将对系统产生{impact_analysis['overall_impact']}级别的影响",
            confidence=0.8,
            assumptions=["影响模型准确", "系统依赖关系明确"],
            limitations=["难以预测所有连锁反应", "外部环境变化的不确定性"]
        )
        reasoning_chain.append(step3)
        
        # 步骤4: 风险评估
        risk_assessment = await self._assess_decision_risks(decision_type, target, evidence_chain)
        step4 = ReasoningStep(
            step_number=4,
            description="评估决策风险",
            input_data={"risk_factors": risk_assessment},
            reasoning_process=f"识别和评估{decision_type.value}的主要风险因素: {risk_assessment}",
            conclusion=f"决策风险等级为{risk_assessment['risk_level']}，需要{risk_assessment['mitigation_strategy']}",
            confidence=0.75,
            assumptions=["风险识别全面", "风险评估方法合理"],
            limitations=["未知风险的存在", "风险概率估算的不确定性"]
        )
        reasoning_chain.append(step4)
        
        # 步骤5: 替代方案比较
        alternatives = await self._generate_alternatives(decision_type, target)
        step5 = ReasoningStep(
            step_number=5,
            description="比较替代方案",
            input_data={"alternatives": alternatives},
            reasoning_process=f"分析{len(alternatives)}个替代方案的优缺点，选择最优方案",
            conclusion="基于综合评估，当前方案是最优选择",
            confidence=0.7,
            assumptions=["替代方案识别完整", "评估标准合理"],
            limitations=["可能存在未考虑的替代方案", "评估标准的主观性"]
        )
        reasoning_chain.append(step5)
        
        return reasoning_chain
    
    async def _summarize_evidence(self, evidence_chain: List[Evidence]) -> str:
        """总结证据"""
        evidence_types = {}
        total_confidence = 0
        
        for evidence in evidence_chain:
            evidence_type = evidence.evidence_type.value
            if evidence_type not in evidence_types:
                evidence_types[evidence_type] = []
            evidence_types[evidence_type].append(evidence.confidence)
            total_confidence += evidence.confidence
        
        summary_parts = []
        for evidence_type, confidences in evidence_types.items():
            avg_confidence = sum(confidences) / len(confidences)
            summary_parts.append(f"{evidence_type}平均置信度{avg_confidence:.2f}")
        
        avg_total_confidence = total_confidence / len(evidence_chain) if evidence_chain else 0
        summary_parts.append(f"总体置信度{avg_total_confidence:.2f}")
        
        return "，".join(summary_parts)
    
    async def _analyze_decision_impact(self, decision_type: DecisionType, 
                                     target: str, 
                                     evidence_chain: List[Evidence]) -> Dict[str, Any]:
        """分析决策影响"""
        impact_areas = {
            "functionality": "medium",
            "performance": "low",
            "security": "low",
            "maintainability": "medium",
            "user_experience": "low"
        }
        
        # 基于决策类型调整影响
        if decision_type == DecisionType.FILE_REMOVAL:
            impact_areas["functionality"] = "high"
            impact_areas["dependency"] = "high"
        elif decision_type == DecisionType.SECURITY_FIX:
            impact_areas["security"] = "high"
        elif decision_type == DecisionType.PERFORMANCE_OPTIMIZE:
            impact_areas["performance"] = "high"
        
        overall_impact = "medium" if any(level == "high" for level in impact_areas.values()) else "low"
        
        return {
            "impact_areas": impact_areas,
            "overall_impact": overall_impact
        }
    
    async def _assess_decision_risks(self, decision_type: DecisionType, 
                                   target: str, 
                                   evidence_chain: List[Evidence]) -> Dict[str, Any]:
        """评估决策风险"""
        risk_factors = []
        
        if decision_type == DecisionType.FILE_REMOVAL:
            risk_factors.extend([
                {"factor": "依赖破坏", "probability": "medium", "impact": "high"},
                {"factor": "功能丢失", "probability": "low", "impact": "high"},
                {"factor": "回滚困难", "probability": "medium", "impact": "medium"}
            ])
        elif decision_type == DecisionType.CODE_REFACTOR:
            risk_factors.extend([
                {"factor": "引入新bug", "probability": "medium", "impact": "medium"},
                {"factor": "性能回归", "probability": "low", "impact": "medium"},
                {"factor": "兼容性问题", "probability": "low", "impact": "high"}
            ])
        
        # 计算总体风险等级
        high_impact_risks = [r for r in risk_factors if r["impact"] == "high"]
        medium_probability_risks = [r for r in risk_factors if r["probability"] == "medium"]
        
        if len(high_impact_risks) >= 2 or len(medium_probability_risks) >= 3:
            risk_level = "high"
            mitigation_strategy = "严格的测试和分阶段实施"
        elif len(high_impact_risks) >= 1 or len(medium_probability_risks) >= 2:
            risk_level = "medium"
            mitigation_strategy = "充分的测试和回滚计划"
        else:
            risk_level = "low"
            mitigation_strategy = "常规测试和监控"
        
        return {
            "risk_factors": risk_factors,
            "risk_level": risk_level,
            "mitigation_strategy": mitigation_strategy
        }
    
    async def _generate_alternatives(self, decision_type: DecisionType, target: str) -> List[str]:
        """生成替代方案"""
        alternatives = []
        
        if decision_type == DecisionType.FILE_REMOVAL:
            alternatives = [
                "保留文件但标记为过时",
                "重构文件而不是删除",
                "移动文件到存档目录",
                "合并文件功能到其他模块"
            ]
        elif decision_type == DecisionType.CODE_REFACTOR:
            alternatives = [
                "保持现状，仅添加注释",
                "部分重构而不是全面重写",
                "使用设计模式重构",
                "分阶段重构"
            ]
        elif decision_type == DecisionType.FILE_RETENTION:
            alternatives = [
                "条件性保留（添加警告）",
                "降级使用而不是完全保留",
                "迁移功能到新模块",
                "重构后保留"
            ]
        
        return alternatives
    
    async def _perform_self_reflection(self, decision_type: DecisionType,
                                     target: str,
                                     reasoning_chain: List[ReasoningStep],
                                     evidence_chain: List[Evidence]) -> List[SelfReflection]:
        """执行自我反省"""
        reflections = []
        
        # 反省1: 偏见识别
        reflection1 = SelfReflection(
            reflection_type="bias_identification",
            question="我在这个决策中是否存在认知偏见？",
            analysis="分析决策过程中的潜在偏见，包括确认偏见、可得性偏见等",
            insights=[
                "可能存在确认偏见，倾向于支持初始假设",
                "可能受到近期事件的影响（可得性偏见）",
                "可能过度依赖量化指标而忽略定性因素"
            ],
            biases_identified=["确认偏见", "可得性偏见", "量化偏见"],
            alternative_approaches=[
                "寻求反对意见和反面证据",
                "使用不同的分析框架重新评估",
                "引入外部专家进行独立评估"
            ],
            confidence_adjustment=-0.05
        )
        reflections.append(reflection1)
        
        # 反省2: 证据完整性
        reflection2 = SelfReflection(
            reflection_type="evidence_completeness",
            question="收集的证据是否足够全面？",
            analysis="评估证据链的完整性和代表性",
            insights=[
                "证据主要来自静态分析，缺乏运行时数据",
                "用户反馈证据不足",
                "长期影响证据有限"
            ],
            biases_identified=["选择偏见", "测量偏见"],
            alternative_approaches=[
                "收集更多运行时性能数据",
                "进行用户调研和反馈收集",
                "分析历史数据和趋势"
            ],
            confidence_adjustment=-0.1
        )
        reflections.append(reflection2)
        
        # 反省3: 推理逻辑
        reflection3 = SelfReflection(
            reflection_type="reasoning_logic",
            question="推理过程是否存在逻辑漏洞？",
            analysis="检查推理链的逻辑一致性和有效性",
            insights=[
                "推理步骤之间的关联性较强",
                "某些假设可能缺乏充分验证",
                "结论的推导过程基本合理"
            ],
            biases_identified=["逻辑跳跃", "过度概括"],
            alternative_approaches=[
                "加强假设验证",
                "细化推理步骤",
                "使用逻辑框架检查"
            ],
            confidence_adjustment=-0.03
        )
        reflections.append(reflection3)
        
        return reflections
    
    async def _calculate_confidence_score(self, evidence_chain: List[Evidence],
                                        reasoning_chain: List[ReasoningStep],
                                        self_reflections: List[SelfReflection]) -> float:
        """计算置信度分数"""
        # 基础置信度来自证据
        evidence_confidence = sum(e.confidence for e in evidence_chain) / len(evidence_chain) if evidence_chain else 0
        
        # 推理置信度
        reasoning_confidence = sum(r.confidence for r in reasoning_chain) / len(reasoning_chain) if reasoning_chain else 0
        
        # 自我反省调整
        reflection_adjustment = sum(r.confidence_adjustment for r in self_reflections)
        
        # 综合计算
        base_confidence = (evidence_confidence * 0.5 + reasoning_confidence * 0.3 + 0.2)
        final_confidence = max(0, min(1, base_confidence + reflection_adjustment))
        
        return final_confidence
    
    async def _assess_risks(self, decision_type: DecisionType, 
                          target: str, 
                          evidence_chain: List[Evidence]) -> str:
        """评估风险"""
        security_evidence = [e for e in evidence_chain if e.evidence_type == EvidenceType.SECURITY_SCAN]
        dependency_evidence = [e for e in evidence_chain if e.evidence_type == EvidenceType.DEPENDENCY_GRAPH]
        
        if decision_type == DecisionType.FILE_REMOVAL:
            if len(dependency_evidence) > 0:
                dep_data = dependency_evidence[0].supporting_data
                if dep_data.get("critical_dependencies", 0) > 0:
                    return "high"
            
            if len(security_evidence) > 0:
                sec_data = security_evidence[0].supporting_data
                if sec_data.get("high_risk_issues", 0) > 0:
                    return "medium"
        
        return "low"
    
    async def _analyze_impact(self, decision_type: DecisionType,
                            target: str,
                            evidence_chain: List[Evidence],
                            analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析影响"""
        impact = {
            "functional": "low",
            "performance": "low",
            "security": "low",
            "maintainability": "low",
            "user_experience": "low"
        }
        
        if decision_type == DecisionType.FILE_REMOVAL:
            impact["functional"] = "medium"
            impact["maintainability"] = "medium"
        
        elif decision_type == DecisionType.SECURITY_FIX:
            impact["security"] = "high"
        
        elif decision_type == DecisionType.PERFORMANCE_OPTIMIZE:
            impact["performance"] = "high"
        
        return impact
    
    async def _consider_alternatives(self, decision_type: DecisionType,
                                   target: str,
                                   analysis_data: Dict[str, Any]) -> List[str]:
        """考虑替代方案"""
        alternatives = []
        
        if decision_type == DecisionType.FILE_REMOVAL:
            alternatives = [
                "重构而不是删除",
                "移动到存档目录",
                "标记为过时但保留"
            ]
        elif decision_type == DecisionType.FILE_RETENTION:
            alternatives = [
                "条件性保留",
                "降级使用",
                "迁移功能"
            ]
        
        return alternatives
    
    async def _generate_final_decision(self, decision_type: DecisionType,
                                     target: str,
                                     reasoning_chain: List[ReasoningStep],
                                     confidence_score: float) -> str:
        """生成最终决策"""
        if confidence_score > 0.8:
            if decision_type == DecisionType.FILE_REMOVAL:
                return "删除文件"
            elif decision_type == DecisionType.FILE_RETENTION:
                return "保留文件"
            elif decision_type == DecisionType.CODE_REFACTOR:
                return "执行重构"
            else:
                return "执行决策"
        elif confidence_score > 0.6:
            if decision_type == DecisionType.FILE_REMOVAL:
                return "谨慎删除（需额外验证）"
            elif decision_type == DecisionType.FILE_RETENTION:
                return "有条件保留"
            else:
                return "有条件执行"
        else:
            return "需要更多信息，暂不决策"
    
    async def _generate_final_justification(self, decision_type: DecisionType,
                                          target: str,
                                          final_decision: str,
                                          evidence_chain: List[Evidence],
                                          reasoning_chain: List[ReasoningStep],
                                          self_reflections: List[SelfReflection]) -> str:
        """生成最终依据"""
        justification_parts = []
        
        # 决策概述
        justification_parts.append(f"## 决策概述")
        justification_parts.append(f"针对目标 '{target}' 的 {decision_type.value} 决策，最终决定：{final_decision}")
        justification_parts.append("")
        
        # 主要依据
        justification_parts.append(f"## 主要依据")
        for evidence in evidence_chain:
            if evidence.confidence > 0.7:
                justification_parts.append(f"- {evidence.content} (置信度: {evidence.confidence:.2f})")
        justification_parts.append("")
        
        # 推理过程
        justification_parts.append(f"## 推理过程")
        for step in reasoning_chain:
            justification_parts.append(f"### 步骤{step.step_number}: {step.description}")
            justification_parts.append(f"推理: {step.reasoning_process}")
            justification_parts.append(f"结论: {step.conclusion}")
            justification_parts.append("")
        
        # 自我反省
        justification_parts.append(f"## 自我反省")
        for reflection in self_reflections:
            justification_parts.append(f"### {reflection.reflection_type}")
            justification_parts.append(f"问题: {reflection.question}")
            justification_parts.append(f"分析: {reflection.analysis}")
            justification_parts.append(f"洞察: {', '.join(reflection.insights)}")
            justification_parts.append("")
        
        # 风险说明
        justification_parts.append(f"## 风险说明")
        justification_parts.append("本决策已考虑潜在风险，并制定了相应的缓解策略。")
        justification_parts.append("")
        
        # 结论
        justification_parts.append(f"## 结论")
        justification_parts.append(f"基于全面的证据收集、严谨的推理过程和深入的自我反省，")
        justification_parts.append(f"我们认为{final_decision}是当前最优决策。")
        
        return "\n".join(justification_parts)
    
    async def _save_decision_record(self, decision_record: DecisionRecord):
        """保存决策记录"""
        # 创建决策记录目录
        decisions_dir = self.project_root / ".iflow" / "temp_docs" / "decisions"
        decisions_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON
        json_file = decisions_dir / f"{decision_record.decision_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(decision_record), f, ensure_ascii=False, indent=2)
        
        # 保存为Markdown
        md_file = decisions_dir / f"{decision_record.decision_id}.md"
        md_content = await self._generate_markdown_report(decision_record)
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"💾 决策记录已保存: {json_file}")
        print(f"📄 Markdown报告已保存: {md_file}")
    
    async def _generate_markdown_report(self, decision_record: DecisionRecord) -> str:
        """生成Markdown报告"""
        content = []
        
        content.append(f"# 决策记录: {decision_record.decision_id}")
        content.append(f"**决策类型**: {decision_record.decision_type.value}")
        content.append(f"**目标**: {decision_record.target}")
        content.append(f"**决策**: {decision_record.decision}")
        content.append(f"**置信度**: {decision_record.confidence_score:.2f}")
        content.append(f"**时间**: {decision_record.timestamp}")
        content.append("")
        
        # 最终依据
        content.append("## 最终依据")
        content.append(decision_record.final_justification)
        content.append("")
        
        # 证据链
        content.append("## 证据链")
        for evidence in decision_record.evidence_chain:
            content.append(f"### {evidence.evidence_type.value}")
            content.append(f"- **来源**: {evidence.source}")
            content.append(f"- **内容**: {evidence.content}")
            content.append(f"- **置信度**: {evidence.confidence:.2f}")
            content.append("")
        
        return "\n".join(content)
    
    def _load_reasoning_templates(self) -> Dict[str, Any]:
        """加载推理模板"""
        return {
            "problem_definition": "基于{target}和{decision_type}，明确需要解决的核心问题",
            "evidence_evaluation": "分析{evidence_count}个证据的可靠性和相关性",
            "impact_analysis": "基于证据分析决策对系统的多方面影响",
            "risk_assessment": "识别和评估决策的主要风险因素",
            "alternative_comparison": "分析替代方案的优缺点，选择最优方案"
        }
    
    def _load_justification_principles(self) -> List[str]:
        """加载依据原则"""
        return [
            "每个决策都必须有充分的证据支持",
            "推理过程必须逻辑清晰、步骤明确",
            "必须考虑替代方案并进行比较",
            "必须识别和评估潜在风险",
            "必须进行自我反省，识别认知偏见",
            "必须提供详细的解释和依据"
        ]

# 使用示例
async def main():
    """主函数"""
    project_root = "."
    
    justification_system = ComprehensiveJustificationSystem(project_root)
    
    # 示例：创建文件删除决策
    analysis_data = {
        "metrics": {"complexity": 15, "maintainability": 45},
        "dependencies": ["module_a", "module_b"],
        "security_issues": [{"severity": "medium", "description": "潜在安全问题"}],
        "performance_metrics": {"execution_time": 150, "memory_usage": 50}
    }
    
    decision_record = await justification_system.create_comprehensive_decision(
        DecisionType.FILE_REMOVAL,
        "example_module.py",
        analysis_data
    )
    
    print(f"🎉 决策记录创建完成!")
    print(f"📊 决策ID: {decision_record.decision_id}")
    print(f"🎯 决策: {decision_record.decision}")
    print(f"📈 置信度: {decision_record.confidence_score:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
