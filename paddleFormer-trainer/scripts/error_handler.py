#!/usr/bin/env python3
"""
Error Handler - Agent驱动的错误检测与修复模块

从"代码驱动"重构为"Agent驱动"架构:
- Skill提供错误信号检测和上下文信息
- Agent基于上下文推理决策修复策略
- 显式暴露不可修复错误，不静默处理
"""

import os
import re
import shutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class ErrorType(Enum):
    """错误类型枚举"""
    UNKNOWN = "unknown"
    MISSING_PARAMETER = "missing_parameter"
    INVALID_CONFIG = "invalid_config"
    OUT_OF_MEMORY = "out_of_memory"
    COMMUNICATION_TIMEOUT = "communication_timeout"
    NETWORK_INTERFACE = "network_interface"
    OPERATOR_NOT_SUPPORTED = "operator_not_supported"
    RUNTIME_ERROR = "runtime_error"


class RepairStrategy(Enum):
    """修复策略枚举"""
    FROM_REFERENCE = "from_reference"           # 从参考配置获取
    USE_DEFAULT = "use_default"                 # 使用默认值
    REDUCE_BATCH_SIZE = "reduce_batch_size"     # 减小batch size
    INCREASE_TIMEOUT = "increase_timeout"       # 增加超时
    AUTO_DETECT_INTERFACE = "auto_detect"       # 自动检测接口
    COMMENT_OUT_OPERATOR = "comment_out"        # 注释掉算子
    CANNOT_FIX = "cannot_fix"                   # 无法修复


@dataclass
class ErrorInfo:
    """错误信息数据结构"""
    error_type: ErrorType
    error_message: str
    repairable: bool
    confidence: float = 0.8
    extracted_params: Dict[str, Any] = field(default_factory=dict)
    suggested_strategy: Optional[RepairStrategy] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepairContext:
    """修复决策上下文"""
    error_info: ErrorInfo
    current_config: Dict[str, Any]
    repair_attempt_count: int
    max_repair_attempts: int
    reference_values: Dict[str, Any]
    available_actions: List[RepairStrategy]


class ErrorHandler:
    """
    错误处理器 - Agent驱动架构
    
    职责分离:
    - Skill层: 错误信号检测、上下文组装、修复执行
    - Agent层: 推理决策(在调用方实现)
    """
    
    # 错误检测模式
    ERROR_PATTERNS = {
        ErrorType.OUT_OF_MEMORY: [
            re.compile(r"out\s+of\s+memory", re.IGNORECASE),
            re.compile(r"XPU\s+OOM", re.IGNORECASE),
            re.compile(r"allocate\s+memory\s+failed", re.IGNORECASE),
            re.compile(r"memory\s+allocation\s+failed", re.IGNORECASE),
        ],
        ErrorType.COMMUNICATION_TIMEOUT: [
            re.compile(r"BKCL\s+error", re.IGNORECASE),
            re.compile(r"NCCL\s+error", re.IGNORECASE),
            re.compile(r"communication\s+timeout", re.IGNORECASE),
            re.compile(r"socket\s+timeout", re.IGNORECASE),
        ],
        ErrorType.MISSING_PARAMETER: [
            re.compile(r"KeyError[:\s]+['\"](\w+)['\"]"),
            re.compile(r"missing\s+required\s+parameter[:\s]+(\w+)", re.IGNORECASE),
            re.compile(r"config\s+key\s+['\"](\w+)['\"]\s+not\s+found", re.IGNORECASE),
        ],
        ErrorType.INVALID_CONFIG: [
            re.compile(r"invalid\s+config", re.IGNORECASE),
            re.compile(r"invalid\s+value\s+for\s+(\w+)", re.IGNORECASE),
            re.compile(r"No\s+such\s+file\s+or\s+directory"),
        ],
        ErrorType.NETWORK_INTERFACE: [
            re.compile(r"network\s+interface", re.IGNORECASE),
            re.compile(r"socket\s+ifname", re.IGNORECASE),
            re.compile(r"cannot\s+bind\s+to\s+interface", re.IGNORECASE),
        ],
        ErrorType.OPERATOR_NOT_SUPPORTED: [
            re.compile(r"not\s+supported", re.IGNORECASE),
            re.compile(r"Op\s+not\s+implemented", re.IGNORECASE),
            re.compile(r"kernel\s+not\s+found", re.IGNORECASE),
        ],
        ErrorType.RUNTIME_ERROR: [
            re.compile(r"RuntimeError"),
            re.compile(r"Segmentation\s+fault", re.IGNORECASE),
            re.compile(r"AssertionError"),
            re.compile(r"CUDA\s+error", re.IGNORECASE),
            re.compile(r"XPU\s+error", re.IGNORECASE),
        ],
    }
    
    # 训练启动成功检测 - 移除简单的loss正则，改用语义判断
    # 仅保留明确的训练启动指示器
    TRAINING_START_PATTERNS = [
        re.compile(r"Starting\s+training", re.IGNORECASE),
        re.compile(r"Begin\s+training", re.IGNORECASE),
    ]
    
    def __init__(self, reference_config_path: str = None):
        """
        初始化错误处理器
        
        Args:
            reference_config_path: 参考配置文件路径
        """
        self.reference_config_path = reference_config_path or self._get_default_reference_path()
        self.repair_history: List[Dict[str, Any]] = []
        
    def _get_default_reference_path(self) -> str:
        """获取默认参考配置路径"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, '..', 'reference_configs', 'xpu_reference.yaml')
    
    def detect_error_signals(self, log_content: str) -> List[Dict[str, Any]]:
        """
        检测错误信号
        
        Args:
            log_content: 日志内容
            
        Returns:
            检测到的错误信号列表
        """
        signals = []
        
        for error_type, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(log_content)
                if match:
                    signal = {
                        "error_type": error_type,
                        "pattern": pattern.pattern[:50],
                        "matched_text": match.group()[:100],
                        "extracted_params": match.groups() if match.groups() else [],
                    }
                    signals.append(signal)
                    break  # 每种错误类型只记录一次
        
        return signals
    
    def analyze_error(self, log_content: str) -> ErrorInfo:
        """
        分析错误日志，生成错误信息
        
        Args:
            log_content: 日志内容
            
        Returns:
            错误信息结构
        """
        signals = self.detect_error_signals(log_content)
        
        if not signals:
            return ErrorInfo(
                error_type=ErrorType.UNKNOWN,
                error_message="未检测到已知错误模式",
                repairable=False,
                confidence=0.5,
            )
        
        # 取第一个检测到的信号（优先级最高）
        primary_signal = signals[0]
        error_type = primary_signal["error_type"]
        
        # 判断可修复性
        repairable, strategy = self._assess_repairability(error_type, signals)
        
        # 构建错误消息
        error_msg = f"{error_type.value}: {primary_signal['matched_text']}"
        
        return ErrorInfo(
            error_type=error_type,
            error_message=error_msg,
            repairable=repairable,
            confidence=0.85 if len(signals) > 0 else 0.6,
            extracted_params={"signals": signals},
            suggested_strategy=strategy,
            context={"signal_count": len(signals)}
        )
    
    def _assess_repairability(self, error_type: ErrorType, signals: List[Dict]) -> Tuple[bool, RepairStrategy]:
        """
        评估错误可修复性（Skill层初步评估）
        
        Args:
            error_type: 错误类型
            signals: 错误信号列表
            
        Returns:
            (是否可修复, 建议策略)
        """
        # 可修复错误映射
        repairable_map = {
            ErrorType.MISSING_PARAMETER: (True, RepairStrategy.FROM_REFERENCE),
            ErrorType.INVALID_CONFIG: (True, RepairStrategy.USE_DEFAULT),
            ErrorType.OUT_OF_MEMORY: (True, RepairStrategy.REDUCE_BATCH_SIZE),
            ErrorType.COMMUNICATION_TIMEOUT: (True, RepairStrategy.INCREASE_TIMEOUT),
            ErrorType.NETWORK_INTERFACE: (True, RepairStrategy.AUTO_DETECT_INTERFACE),
            ErrorType.OPERATOR_NOT_SUPPORTED: (True, RepairStrategy.COMMENT_OUT_OPERATOR),
            # 不可修复错误
            ErrorType.RUNTIME_ERROR: (False, RepairStrategy.CANNOT_FIX),
            ErrorType.UNKNOWN: (False, RepairStrategy.CANNOT_FIX),
        }
        
        return repairable_map.get(error_type, (False, RepairStrategy.CANNOT_FIX))
    
    def get_error_context(
        self,
        log_content: str,
        config_path: str,
        model_name: str,
        repair_attempt: int = 0,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        获取错误处理完整上下文（供Agent使用）
        
        Args:
            log_content: 日志内容
            config_path: 配置文件路径
            model_name: 模型名称
            repair_attempt: 当前修复尝试次数
            max_attempts: 最大修复尝试次数
            
        Returns:
            完整上下文信息
        """
        error_info = self.analyze_error(log_content)
        current_config = self._load_config(config_path) if os.path.exists(config_path) else {}
        reference_config = self._load_reference_config()
        
        # 提取相关配置参数
        relevant_params = self._extract_relevant_params(error_info, current_config)
        
        context = {
            "error_detection": {
                "signals_detected": error_info.error_type != ErrorType.UNKNOWN,
                "error_info": {
                    "type": error_info.error_type.value,
                    "message": error_info.error_message,
                    "repairable": error_info.repairable,
                    "confidence": error_info.confidence,
                    "extracted_params": error_info.extracted_params,
                    "suggested_strategy": error_info.suggested_strategy.value if error_info.suggested_strategy else None,
                },
                "log_snippet": log_content[-2000:] if len(log_content) > 2000 else log_content,
            },
            "current_state": {
                "config_path": config_path,
                "model_name": model_name,
                "current_config": current_config,
                "relevant_params": relevant_params,
                "repair_attempt_count": repair_attempt,
                "max_repair_attempts": max_attempts,
            },
            "reference_values": reference_config,
            "available_repair_actions": [s.value for s in RepairStrategy],
        }
        
        return context
    
    def _extract_relevant_params(self, error_info: ErrorInfo, config: Dict) -> Dict[str, Any]:
        """提取与错误相关的配置参数"""
        relevant = {}
        
        # 从错误信号中提取关键参数
        signals = error_info.extracted_params.get("signals", [])
        for signal in signals:
            params = signal.get("extracted_params", [])
            for param in params:
                if isinstance(param, str) and param in config:
                    relevant[param] = config[param]
        
        # 根据错误类型添加相关参数
        if error_info.error_type == ErrorType.OUT_OF_MEMORY:
            relevant.update({
                "per_device_train_batch_size": config.get("per_device_train_batch_size", "N/A"),
                "gradient_accumulation_steps": config.get("gradient_accumulation_steps", "N/A"),
            })
        elif error_info.error_type == ErrorType.COMMUNICATION_TIMEOUT:
            relevant.update({
                "bkcl_timeout": config.get("bkcl_timeout", "N/A"),
            })
        
        return relevant
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            return {"_load_error": str(e)}
    
    def _load_reference_config(self) -> Dict[str, Any]:
        """加载参考配置"""
        return self._load_config(self.reference_config_path)
    
    def execute_repair(
        self,
        config_path: str,
        strategy: RepairStrategy,
        changes: Dict[str, Any],
        comment_out_fields: List[str] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        执行修复（Skill层执行Agent决策）
        
        Args:
            config_path: 配置文件路径
            strategy: 修复策略
            changes: 配置修改
            comment_out_fields: 需要注释掉的字段
            
        Returns:
            (是否成功, 新配置路径, 详细信息)
        """
        comment_out_fields = comment_out_fields or []
        
        try:
            # 读取原配置
            with open(config_path, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
            
            # 创建备份
            backup_path = f"{config_path}.backup.{self._timestamp()}"
            shutil.copy2(config_path, backup_path)
            
            # 应用修改
            modified_lines = self._apply_changes(
                original_lines, changes, comment_out_fields
            )
            
            # 保存新配置
            new_config_path = config_path
            with open(new_config_path, 'w', encoding='utf-8') as f:
                f.writelines(modified_lines)
            
            # 记录修复历史
            repair_record = {
                "timestamp": self._timestamp(),
                "strategy": strategy.value,
                "changes": changes,
                "comment_out_fields": comment_out_fields,
                "backup_path": backup_path,
                "config_path": new_config_path,
            }
            self.repair_history.append(repair_record)
            
            details = {
                "changes": changes,
                "comment_out_fields": comment_out_fields,
                "backup_path": backup_path,
                "repair_record": repair_record,
            }
            
            return True, new_config_path, details
            
        except Exception as e:
            return False, config_path, {"error": str(e)}
    
    def _apply_changes(
        self,
        lines: List[str],
        changes: Dict[str, Any],
        comment_out_fields: List[str]
    ) -> List[str]:
        """应用配置修改"""
        result = []
        modified_keys = set()
        
        for line in lines:
            original_line = line
            modified = False
            
            # 检查是否需要注释掉
            for field in comment_out_fields:
                if line.strip().startswith(f"{field}:"):
                    line = f"# {line}"
                    modified = True
                    break
            
            # 检查是否需要修改值
            if not modified:
                for key, value in changes.items():
                    if line.strip().startswith(f"{key}:"):
                        # 保持缩进
                        indent = len(line) - len(line.lstrip())
                        line = " " * indent + f"{key}: {value}\n"
                        modified_keys.add(key)
                        modified = True
                        break
            
            result.append(line)
        
        # 添加新增的配置项
        for key, value in changes.items():
            if key not in modified_keys:
                result.append(f"{key}: {value}\n")
        
        return result
    
    def _timestamp(self) -> str:
        """生成时间戳"""
        from datetime import datetime
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def get_repair_suggestions(self, error_info: ErrorInfo) -> List[str]:
        """
        获取修复建议（供Agent或用户使用）
        
        Args:
            error_info: 错误信息
            
        Returns:
            建议列表
        """
        suggestions_map = {
            ErrorType.MISSING_PARAMETER: [
                "检查配置文件是否缺少必需字段",
                "参考xpu_reference.yaml补全缺失参数",
                "确认模型配置与训练任务匹配",
            ],
            ErrorType.INVALID_CONFIG: [
                "检查配置参数值是否合法",
                "确认文件路径存在且可访问",
                "验证配置格式是否正确",
            ],
            ErrorType.OUT_OF_MEMORY: [
                "减小per_device_train_batch_size",
                "增大gradient_accumulation_steps",
                "减少模型并行度",
            ],
            ErrorType.COMMUNICATION_TIMEOUT: [
                "增加bkcl_timeout值",
                "检查网络连接稳定性",
                "确认BKCL配置正确",
            ],
            ErrorType.RUNTIME_ERROR: [
                "检查Paddle/PaddleFormers版本兼容性",
                "确认XPU驱动和环境配置正确",
                "查看完整错误堆栈定位问题",
            ],
            ErrorType.OPERATOR_NOT_SUPPORTED: [
                "注释掉不支持的算子配置",
                "更新PaddleFormers到最新版本",
                "联系XPU支持团队确认算子支持状态",
            ],
            ErrorType.UNKNOWN: [
                "查看完整训练日志",
                "检查环境配置",
                "尝试手动复现问题",
            ],
        }
        
        return suggestions_map.get(error_info.error_type, ["未知错误，建议人工分析"])
    
    def is_training_started(self, log_content: str) -> bool:
        """
        检测训练是否已开始（基于上下文的语义判断）
        
        修改：不再依赖简单正则匹配loss，而是基于上下文判断是否是真实训练输出
        
        Args:
            log_content: 日志内容
            
        Returns:
            是否已检测到训练开始信号
        """
        lines = log_content.strip().split('\n') if log_content else []
        
        # 检查最近20行是否包含训练循环特征
        has_step = False
        has_number = False
        has_training_kw = False
        
        for line in lines[-20:]:
            line_lower = line.lower()
            # 检测step指示器
            if re.search(r'(step|iteration|batch)\s*[=:]?\s*\d+', line_lower):
                has_step = True
            # 检测数字（可能是loss）
            if re.search(r'[\s:](\d+\.\d+)([,\s]|$)', line):
                has_number = True
            # 检测训练关键词
            if any(kw in line_lower for kw in ['train', 'epoch', 'forward', 'backward']):
                has_training_kw = True
        
        # 关键判断：同时有step和数字，或者有明确的训练启动指示
        if has_step and has_number:
            return True
        
        # 检查明确的训练启动指示
        for pattern in self.TRAINING_START_PATTERNS:
            if pattern.search(log_content):
                return True
        
        return False
    
    def get_repair_history(self) -> List[Dict[str, Any]]:
        """获取修复历史"""
        return self.repair_history


# 便捷函数
def analyze_training_error(
    log_content: str,
    config_path: str = None,
    model_name: str = "unknown"
) -> Dict[str, Any]:
    """
    分析训练错误的便捷函数
    
    Args:
        log_content: 错误日志内容
        config_path: 配置文件路径
        model_name: 模型名称
        
    Returns:
        分析结果
    """
    handler = ErrorHandler()
    error_info = handler.analyze_error(log_content)
    
    result = {
        "error_type": error_info.error_type.value,
        "error_message": error_info.error_message,
        "repairable": error_info.repairable,
        "confidence": error_info.confidence,
        "suggested_strategy": error_info.suggested_strategy.value if error_info.suggested_strategy else None,
        "suggestions": handler.get_repair_suggestions(error_info),
    }
    
    if config_path:
        context = handler.get_error_context(log_content, config_path, model_name)
        result["context"] = context
    
    return result


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python error_handler.py <log_file> [config_file]")
        sys.exit(1)
    
    log_file = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()
    
    result = analyze_training_error(log_content, config_file)
    
    print("\n错误分析结果:")
    print(f"  类型: {result['error_type']}")
    print(f"  消息: {result['error_message']}")
    print(f"  可修复: {result['repairable']}")
    print(f"  建议策略: {result['suggested_strategy']}")
    print("\n修复建议:")
    for suggestion in result['suggestions']:
        print(f"  - {suggestion}")