#!/usr/bin/env python3
"""
Training Monitor Agent - Agent驱动的训练监控模块

轻量级Agent实现，用于：
1. 主动监控训练日志
2. 推理判断训练状态
3. 动态决策训练是否真正开始
"""

import os
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto


class TrainingStatus(Enum):
    """训练状态枚举"""
    UNKNOWN = auto()           # 未知状态
    INITIALIZING = auto()      # 初始化中
    STARTING = auto()          # 启动中
    RUNNING = auto()           # 正常运行（已检测到loss）
    ERROR = auto()             # 发生错误
    STALLED = auto()           # 卡住/无进展
    TIMEOUT = auto()           # 超时


@dataclass
class LogAnalysisResult:
    """日志分析结果"""
    status: TrainingStatus
    confidence: float          # 置信度 0-1
    evidence: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class MonitorDecision:
    """监控决策结果"""
    should_continue: bool      # 是否继续监控
    is_training_started: bool  # 训练是否已开始
    status: TrainingStatus
    reasoning: str
    recommendations: List[str] = field(default_factory=list)


class TrainingMonitorAgent:
    """
    训练监控Agent
    
    轻量级Agent实现，具备以下能力：
    1. 日志内容分析与推理
    2. 训练状态判断
    3. 错误检测与分类
    4. 动态决策
    """
    
    # Loss检测模式（按优先级排序）
    LOSS_PATTERNS = [
        re.compile(r"loss[:\s]+(\d+\.?\d*)", re.IGNORECASE),
        re.compile(r"train_loss[:\s]+(\d+\.?\d*)", re.IGNORECASE),
        re.compile(r"step[:\s]+\d+.*loss[:\s]+(\d+\.?\d*)", re.IGNORECASE),
        re.compile(r"global_step[:\s]+\d+.*loss", re.IGNORECASE),
    ]
    
    # 错误检测模式
    ERROR_PATTERNS = {
        "out_of_memory": [
            re.compile(r"out\s+of\s+memory", re.IGNORECASE),
            re.compile(r"XPU\s+OOM", re.IGNORECASE),
            re.compile(r"allocate\s+memory\s+failed", re.IGNORECASE),
        ],
        "communication_error": [
            re.compile(r"BKCL\s+error", re.IGNORECASE),
            re.compile(r"NCCL\s+error", re.IGNORECASE),
            re.compile(r"communication\s+timeout", re.IGNORECASE),
        ],
        "config_error": [
            re.compile(r"KeyError[:\s]+'(\w+)'"),
            re.compile(r"invalid\s+config", re.IGNORECASE),
            re.compile(r"No\s+such\s+file\s+or\s+directory"),
        ],
        "runtime_error": [
            re.compile(r"RuntimeError"),
            re.compile(r"Segmentation\s+fault", re.IGNORECASE),
            re.compile(r"AssertionError"),
        ],
        "operator_error": [
            re.compile(r"not\s+supported", re.IGNORECASE),
            re.compile(r"Op\s+not\s+implemented", re.IGNORECASE),
        ],
    }
    
    # 初始化/启动指示模式
    INITIALIZATION_PATTERNS = [
        re.compile(r"Loading\s+checkpoint", re.IGNORECASE),
        re.compile(r"Initializing", re.IGNORECASE),
        re.compile(r"Loading\s+model", re.IGNORECASE),
        re.compile(r"Preparing\s+data", re.IGNORECASE),
        re.compile(r"Start\s+training", re.IGNORECASE),
        re.compile(r"Begin\s+training", re.IGNORECASE),
    ]
    
    def __init__(self, timeout: int = 300, poll_interval: int = 5):
        """
        初始化监控Agent
        
        Args:
            timeout: 监控超时时间（秒）
            poll_interval: 轮询间隔（秒）
        """
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.observations: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        
    def analyze_log_content(self, log_content: str, timestamp: Optional[float] = None) -> LogAnalysisResult:
        """
        Agent推理：分析日志内容，判断训练状态
        
        Args:
            log_content: 日志内容
            timestamp: 时间戳
            
        Returns:
            分析结果
        """
        if timestamp is None:
            timestamp = time.time()
            
        evidence = []
        metrics = {}
        
        # 1. 检测Loss输出（训练已开始的关键证据）
        loss_detected = False
        loss_values = []
        for pattern in self.LOSS_PATTERNS:
            matches = pattern.findall(log_content)
            if matches:
                loss_detected = True
                loss_values.extend([float(m) if isinstance(m, str) else float(m[0]) for m in matches])
                evidence.append(f"检测到Loss模式: {pattern.pattern[:30]}...")
                
        if loss_detected:
            metrics["loss_values"] = loss_values[:5]  # 保留前5个
            metrics["loss_count"] = len(loss_values)
            return LogAnalysisResult(
                status=TrainingStatus.RUNNING,
                confidence=0.95,
                evidence=evidence,
                metrics=metrics,
                reasoning=f"检测到{len(loss_values)}个loss值，训练已正常运行"
            )
        
        # 2. 检测错误
        for error_type, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(log_content)
                if match:
                    evidence.append(f"检测到错误: {error_type} - {match.group()[:50]}")
                    metrics["error_type"] = error_type
                    metrics["error_match"] = match.group()
                    return LogAnalysisResult(
                        status=TrainingStatus.ERROR,
                        confidence=0.9,
                        evidence=evidence,
                        metrics=metrics,
                        reasoning=f"检测到{error_type}错误，训练启动失败"
                    )
        
        # 3. 检测初始化阶段
        init_detected = False
        for pattern in self.INITIALIZATION_PATTERNS:
            if pattern.search(log_content):
                init_detected = True
                evidence.append(f"检测到初始化: {pattern.pattern[:30]}...")
                
        if init_detected:
            return LogAnalysisResult(
                status=TrainingStatus.INITIALIZING,
                confidence=0.7,
                evidence=evidence,
                metrics=metrics,
                reasoning="检测到初始化活动，训练正在启动中"
            )
        
        # 4. 无任何已知模式
        return LogAnalysisResult(
            status=TrainingStatus.UNKNOWN,
            confidence=0.5,
            evidence=["未检测到已知模式"],
            metrics={"log_length": len(log_content)},
            reasoning="日志中未检测到loss、错误或初始化信号"
        )
    
    def make_decision(
        self,
        current_analysis: LogAnalysisResult,
        elapsed_time: float,
        observation_history: List[Dict[str, Any]]
    ) -> MonitorDecision:
        """
        Agent决策：基于分析结果决定是否继续监控
        
        Args:
            current_analysis: 当前分析结果
            elapsed_time: 已用时间
            observation_history: 历史观察记录
            
        Returns:
            决策结果
        """
        status = current_analysis.status
        recommendations = []
        
        # 情况1: 检测到Loss，训练成功开始
        if status == TrainingStatus.RUNNING:
            return MonitorDecision(
                should_continue=False,
                is_training_started=True,
                status=TrainingStatus.RUNNING,
                reasoning=current_analysis.reasoning,
                recommendations=["训练已成功启动，可以停止监控"]
            )
        
        # 情况2: 检测到错误，训练启动失败
        if status == TrainingStatus.ERROR:
            return MonitorDecision(
                should_continue=False,
                is_training_started=False,
                status=TrainingStatus.ERROR,
                reasoning=current_analysis.reasoning,
                recommendations=["检测到错误，建议检查配置并重新启动"]
            )
        
        # 情况3: 超时
        if elapsed_time >= self.timeout:
            # 分析超时前的状态
            if status == TrainingStatus.INITIALIZING:
                reasoning = "训练初始化时间超过预期，可能卡住"
                recommendations = ["检查日志是否有静默失败", "考虑增加超时时间"]
            else:
                reasoning = f"监控超时（{self.timeout}秒），未检测到训练开始信号"
                recommendations = ["检查配置是否正确", "确认模型和数据集可访问"]
                
            return MonitorDecision(
                should_continue=False,
                is_training_started=False,
                status=TrainingStatus.TIMEOUT,
                reasoning=reasoning,
                recommendations=recommendations
            )
        
        # 情况4: 继续监控
        remaining_time = self.timeout - elapsed_time
        progress_pct = min(100, int(elapsed_time / self.timeout * 100))
        
        if status == TrainingStatus.INITIALIZING:
            reasoning = f"训练正在初始化（{progress_pct}%），继续监控..."
            recommendations = [f"剩余等待时间: {remaining_time:.0f}秒"]
        else:
            reasoning = f"等待训练启动信号（{progress_pct}%），继续监控..."
            recommendations = [f"剩余等待时间: {remaining_time:.0f}秒", "建议检查日志输出"]
        
        return MonitorDecision(
            should_continue=True,
            is_training_started=False,
            status=status,
            reasoning=reasoning,
            recommendations=recommendations
        )
    
    def monitor_training_start(
        self,
        log_file_path: str,
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Agent执行：监控训练启动过程
        
        Args:
            log_file_path: 日志文件路径
            callback: 状态更新回调函数(status_msg: str)
            
        Returns:
            监控结果
        """
        self.start_time = time.time()
        self.observations = []
        last_log_size = 0
        
        def notify(msg: str):
            if callback:
                callback(msg)
        
        notify(f"🤖 Agent开始监控训练启动...")
        notify(f"   日志文件: {log_file_path}")
        notify(f"   超时设置: {self.timeout}秒")
        
        while True:
            elapsed = time.time() - self.start_time
            
            # 读取日志内容
            log_content = ""
            if os.path.exists(log_file_path):
                try:
                    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        # 只读取新增内容
                        f.seek(last_log_size)
                        new_content = f.read()
                        log_content = new_content
                        last_log_size = f.tell()
                except Exception as e:
                    notify(f"   读取日志失败: {e}")
            
            # Agent分析日志
            analysis = self.analyze_log_content(log_content if log_content else "")
            
            # 记录观察
            self.observations.append({
                "timestamp": time.time(),
                "elapsed": elapsed,
                "status": analysis.status.name,
                "evidence": analysis.evidence,
            })
            
            # Agent决策
            decision = self.make_decision(analysis, elapsed, self.observations)
            
            # 输出推理过程（仅在状态变化或重要时刻）
            if log_content or not decision.should_continue:
                notify(f"\n📊 Agent分析 [{elapsed:.1f}s]")
                notify(f"   状态: {analysis.status.name} (置信度: {analysis.confidence:.0%})")
                notify(f"   推理: {decision.reasoning}")
                if analysis.evidence:
                    for ev in analysis.evidence[:2]:  # 只显示前2条证据
                        notify(f"   证据: {ev}")
            
            # 检查是否应该停止
            if not decision.should_continue:
                result = {
                    "success": decision.is_training_started,
                    "status": decision.status.name.lower(),
                    "elapsed_time": elapsed,
                    "reasoning": decision.reasoning,
                    "recommendations": decision.recommendations,
                    "observations": self.observations,
                    "final_analysis": {
                        "status": analysis.status.name,
                        "confidence": analysis.confidence,
                        "metrics": analysis.metrics,
                    }
                }
                
                if decision.is_training_started:
                    notify(f"\n✅ Agent判定: 训练已成功启动！")
                    notify(f"   启动耗时: {elapsed:.1f}秒")
                    if analysis.metrics.get("loss_values"):
                        notify(f"   检测到Loss: {analysis.metrics['loss_values'][:3]}")
                else:
                    notify(f"\n❌ Agent判定: 训练启动失败")
                    notify(f"   原因: {decision.reasoning}")
                    for rec in decision.recommendations:
                        notify(f"   建议: {rec}")
                
                return result
            
            # 等待下一次轮询
            time.sleep(self.poll_interval)
    
    def get_status_report(self) -> str:
        """生成Agent监控报告"""
        if not self.observations:
            return "暂无监控记录"
        
        lines = ["\n📋 Agent监控报告", "=" * 40]
        lines.append(f"总观察次数: {len(self.observations)}")
        lines.append(f"总监控时长: {self.observations[-1]['elapsed']:.1f}秒")
        
        # 统计各状态出现次数
        status_counts = {}
        for obs in self.observations:
            status = obs["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        lines.append("\n状态分布:")
        for status, count in sorted(status_counts.items()):
            lines.append(f"  - {status}: {count}次")
        
        return "\n".join(lines)


# 便捷函数
def monitor_training_with_agent(
    log_file_path: str,
    timeout: int = 300,
    poll_interval: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    使用Agent监控训练启动
    
    Args:
        log_file_path: 日志文件路径
        timeout: 超时时间
        poll_interval: 轮询间隔
        verbose: 是否输出详细信息
        
    Returns:
        监控结果
    """
    agent = TrainingMonitorAgent(timeout=timeout, poll_interval=poll_interval)
    
    def print_callback(msg: str):
        if verbose:
            print(msg)
    
    return agent.monitor_training_start(log_file_path, callback=print_callback)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python training_monitor_agent.py <log_file_path>")
        sys.exit(1)
    
    log_path = sys.argv[1]
    result = monitor_training_with_agent(log_path, timeout=60)
    
    print("\n" + "=" * 40)
    print("监控结果:")
    print(f"  成功: {result['success']}")
    print(f"  状态: {result['status']}")
    print(f"  耗时: {result['elapsed_time']:.1f}秒")
    print(f"  推理: {result['reasoning']}")