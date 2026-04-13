#!/usr/bin/env python3
"""
错误检测与自修复处理器 (Agent 驱动版)

此脚本为 Agent 提供错误检测与修复的基础能力。
复杂的修复决策逻辑由 AI 根据 SKILL.md 中的规则自主实现。
"""

import os
import re
import yaml
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum


class ErrorType(Enum):
    """错误类型枚举 - 作为信号供 Agent 参考"""
    MISSING_PARAMETER = "missing_parameter"
    INVALID_CONFIG = "invalid_config"
    OUT_OF_MEMORY = "out_of_memory"
    COMMUNICATION_TIMEOUT = "communication_timeout"
    NETWORK_INTERFACE = "network_interface"
    OPERATOR_NOT_SUPPORTED = "operator_not_supported"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN = "unknown"


class RepairStrategy(Enum):
    """修复策略枚举 - 供 Agent 选择"""
    FROM_REFERENCE = "from_reference"
    USE_DEFAULT = "use_default"
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    INCREASE_TIMEOUT = "increase_timeout"
    AUTO_DETECT_INTERFACE = "auto_detect_interface"
    COMMENT_OUT_OPERATOR = "comment_out_operator"
    CANNOT_FIX = "cannot_fix"


class ErrorHandler:
    """错误处理器 - Agent 驱动版

    设计原则:
    1. Skill 提供错误检测信号和修复执行能力
    2. Agent 基于上下文进行修复决策
    3. 不可修复的错误必须显式返回，不得静默处理
    """

    # 错误模式定义 - 仅作为检测信号，不直接决定修复策略
    ERROR_PATTERNS = {
        ErrorType.MISSING_PARAMETER: [
            re.compile(r"KeyError:\s*['\"](\w+)['\"]"),
            re.compile(r"missing\s+required\s+parameter\s*['\"]?(\w+)['\"]?"),
            re.compile(r"['\"](\w+)['\"]\s+is\s+required"),
            re.compile(r"config\s+key\s+['\"](\w+)['\"]\s+not\s+found"),
        ],
        ErrorType.INVALID_CONFIG: [
            re.compile(r"Invalid\s+.*device\s+specification"),
            re.compile(r"Invalid\s+value\s+for\s+['\"]?(\w+)['\"]?"),
            re.compile(r"['\"]?(\w+)['\"]?\s+must\s+be"),
            re.compile(r"config\s+error:\s*(.+)", re.IGNORECASE),
        ],
        ErrorType.OUT_OF_MEMORY: [
            re.compile(r"XPU\s+OOM", re.IGNORECASE),
            re.compile(r"out\s+of\s+memory", re.IGNORECASE),
            re.compile(r"memory\s+allocation\s+failed", re.IGNORECASE),
            re.compile(r"insufficient\s+memory", re.IGNORECASE),
            re.compile(r"cannot\s+allocate\s+memory", re.IGNORECASE),
        ],
        ErrorType.COMMUNICATION_TIMEOUT: [
            re.compile(r"BKCL.*timeout", re.IGNORECASE),
            re.compile(r"BKCL\s+error.*timeout", re.IGNORECASE),
            re.compile(r"communication.*timeout", re.IGNORECASE),
            re.compile(r"timeout\s+waiting\s+for", re.IGNORECASE),
        ],
        ErrorType.NETWORK_INTERFACE: [
            re.compile(r"BKCL.*socket.*ifname", re.IGNORECASE),
            re.compile(r"network\s+interface.*not\s+found", re.IGNORECASE),
            re.compile(r"cannot\s+find\s+network\s+interface", re.IGNORECASE),
            re.compile(r"BKCL_SOCKET_IFNAME", re.IGNORECASE),
        ],
        ErrorType.OPERATOR_NOT_SUPPORTED: [
            re.compile(r"operator.*not\s+supported", re.IGNORECASE),
            re.compile(r"op.*not\s+implemented", re.IGNORECASE),
            re.compile(r"kernel.*not\s+found", re.IGNORECASE),
            re.compile(r"XPU\s+does\s+not\s+support", re.IGNORECASE),
        ],
        ErrorType.RUNTIME_ERROR: [
            re.compile(r"Segmentation\s+fault", re.IGNORECASE),
            re.compile(r"RuntimeError"),
            re.compile(r"AssertionError"),
            re.compile(r"Floating\s+point\s+exception", re.IGNORECASE),
        ],
    }

    def __init__(
        self,
        reference_config_path: str = None,
        repair_log_path: str = None,
        mapping_rules_path: str = None
    ):
        """初始化错误处理器

        Args:
            reference_config_path: 参考配置文件路径
            repair_log_path: 修复日志文件路径
            mapping_rules_path: 映射规则文件路径（包含错误处理规则）
        """
        # 参考配置路径
        if reference_config_path is None:
            reference_config_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'reference_configs',
                'xpu_reference.yaml'
            )
        self.reference_config_path = reference_config_path
        self.reference_config = self._load_reference_config()

        # 修复日志路径
        if repair_log_path is None:
            repair_log_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'repair_log.txt'
            )
        self.repair_log_path = repair_log_path

        # 映射规则路径（包含错误处理规则）
        if mapping_rules_path is None:
            mapping_rules_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'templates',
                'mapping_rules.yaml'
            )
        self.mapping_rules_path = mapping_rules_path
        self.error_rules = self._load_error_rules()

        # 修复计数
        self.repair_count = 0
        self.max_repair_attempts = 3

    def _load_reference_config(self) -> Dict:
        """加载参考配置"""
        try:
            with open(self.reference_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"警告: 无法加载参考配置: {e}")
            return {}

    def _load_error_rules(self) -> Dict:
        """加载错误处理规则"""
        try:
            with open(self.mapping_rules_path, 'r', encoding='utf-8') as f:
                rules = yaml.safe_load(f) or {}
                return rules.get('error_handling', {})
        except Exception as e:
            print(f"警告: 无法加载错误处理规则: {e}")
            return {}

    def detect_error_signals(self, log_content: str) -> Dict[str, Any]:
        """检测错误信号 - 为 Agent 提供初步分析

        Args:
            log_content: 错误日志内容

        Returns:
            错误信号检测结果，供 Agent 进一步分析
        """
        signals = []

        # 检查每种错误类型的匹配模式
        for error_type, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                matches = pattern.finditer(log_content)
                for match in matches:
                    signal = {
                        "error_type": error_type.value,
                        "matched_pattern": pattern.pattern,
                        "matched_text": match.group(0),
                        "position": match.span(),
                        "extracted_params": list(match.groups()) if match.groups() else [],
                    }
                    signals.append(signal)

        # 去重并按位置排序
        seen = set()
        unique_signals = []
        for s in signals:
            key = (s["error_type"], s["position"][0])
            if key not in seen:
                seen.add(key)
                unique_signals.append(s)

        unique_signals.sort(key=lambda x: x["position"][0])

        return {
            "signals_detected": len(unique_signals) > 0,
            "signals": unique_signals,
            "log_snippet": log_content[-2000:] if len(log_content) > 2000 else log_content,
        }

    def get_error_context(
        self,
        log_content: str,
        config_path: str = None,
        model_name: str = None
    ) -> Dict[str, Any]:
        """获取错误处理的完整上下文信息，供 Agent 执行修复决策

        这是 Agent 驱动模式的核心方法，提供所有必要的信息和规则，
        由 Agent 基于 SKILL.md 指导完成实际的错误分析与修复决策。

        Args:
            log_content: 错误日志内容
            config_path: 配置文件路径（可选）
            model_name: 模型名称（可选）

        Returns:
            完整的错误处理上下文
        """
        # 1. 检测错误信号
        error_signals = self.detect_error_signals(log_content)

        # 2. 加载当前配置（如果提供）
        current_config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    current_config = yaml.safe_load(f) or {}
            except Exception as e:
                current_config = {"_load_error": str(e)}

        # 3. 提取关键配置参数
        config_params = self._extract_relevant_params(current_config)

        # 4. 构建上下文
        context = {
            "error_detection": error_signals,
            "current_state": {
                "config_path": config_path,
                "model_name": model_name,
                "current_config": current_config,
                "relevant_params": config_params,
                "repair_attempt_count": self.repair_count,
                "max_repair_attempts": self.max_repair_attempts,
            },
            "rules": {
                "error_patterns": {
                    k.value: [p.pattern for p in v]
                    for k, v in self.ERROR_PATTERNS.items()
                },
                "error_handling_rules": self.error_rules,
                "reference_config_keys": list(self.reference_config.keys()),
            },
            "available_repair_actions": {
                "FROM_REFERENCE": "从参考配置补全缺失参数",
                "USE_DEFAULT": "使用默认值修复无效配置",
                "REDUCE_BATCH_SIZE": "减小 batch size 或增加梯度累积",
                "INCREASE_TIMEOUT": "增加通信超时时间",
                "AUTO_DETECT_INTERFACE": "自动检测网络接口",
                "COMMENT_OUT_OPERATOR": "注释掉不支持的算子",
                "CANNOT_FIX": "无法自动修复，需要人工介入",
            },
            "reference_values": self._get_relevant_reference_values(
                error_signals, config_params
            ),
        }

        return context

    def _extract_relevant_params(self, config: Dict) -> Dict[str, Any]:
        """提取与错误修复相关的配置参数"""
        relevant_keys = [
            "device",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "bkcl_timeout",
            "bkcl_socket_ifname",
            "recompute_granularity",
            "max_seq_len",
            "output_dir",
            "model_name_or_path",
        ]
        return {k: config.get(k) for k in relevant_keys if k in config}

    def _get_relevant_reference_values(
        self,
        error_signals: Dict,
        config_params: Dict
    ) -> Dict[str, Any]:
        """获取与当前错误相关的参考值"""
        relevant = {}

        # 根据错误信号决定需要哪些参考值
        signals = error_signals.get("signals", [])
        error_types = {s["error_type"] for s in signals}

        if "missing_parameter" in error_types:
            # 提供常用必需字段的参考值
            relevant["default_required_fields"] = {
                "device": self.reference_config.get("device", "xpu"),
                "bkcl_timeout": self.reference_config.get("bkcl_timeout", 1000),
                "bkcl_socket_ifname": self.reference_config.get("bkcl_socket_ifname", "eth0"),
            }

        if "out_of_memory" in error_types:
            # 提供内存优化相关的参考值
            relevant["memory_optimization"] = {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": self.reference_config.get(
                    "gradient_accumulation_steps", 8
                ),
                "recompute_granularity": "full",
            }

        if "communication_timeout" in error_types:
            # 提供通信相关的参考值
            relevant["communication"] = {
                "bkcl_timeout": min(
                    config_params.get("bkcl_timeout", 1000) * 2,
                    5000
                ),
            }

        return relevant

    def validate_repair_plan(self, repair_plan: Dict[str, Any]) -> Tuple[bool, str]:
        """验证 Agent 生成的修复计划是否合法

        Args:
            repair_plan: Agent 生成的修复计划

        Returns:
            (是否合法, 错误信息)
        """
        if not isinstance(repair_plan, dict):
            return False, "修复计划必须是字典类型"

        # 检查必需的字段
        if "should_repair" not in repair_plan:
            return False, "修复计划必须包含 'should_repair' 字段"

        # 如果不修复，不需要其他字段
        if not repair_plan.get("should_repair"):
            return True, ""

        # 如果要修复，需要指定策略
        if "strategy" not in repair_plan:
            return False, "执行修复时必须指定 'strategy'"

        # 验证策略是否合法
        valid_strategies = [s.value for s in RepairStrategy]
        if repair_plan["strategy"] not in valid_strategies:
            return False, f"非法的策略: {repair_plan['strategy']}"

        # 验证参数变更
        changes = repair_plan.get("config_changes", {})
        if not isinstance(changes, dict):
            return False, "'config_changes' 必须是字典类型"

        # 检查危险操作
        dangerous_keys = ["rm ", "mv ", "system("]
        for key, value in changes.items():
            if isinstance(value, str):
                for dangerous in dangerous_keys:
                    if dangerous in value:
                        return False, f"参数值包含危险操作: {key}={value}"

        return True, ""

    def apply_repair(
        self,
        config_path: str,
        repair_plan: Dict[str, Any],
        error_context: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """执行 Agent 决策的修复计划

        Args:
            config_path: 配置文件路径
            repair_plan: Agent 生成的修复计划
            error_context: 错误上下文（用于日志记录）

        Returns:
            (是否修复成功, 修复后的配置文件路径, 修复详情)
        """
        # 1. 验证修复计划
        valid, error_msg = self.validate_repair_plan(repair_plan)
        if not valid:
            return False, config_path, {
                "error": f"修复计划验证失败: {error_msg}",
                "repair_plan": repair_plan,
            }

        # 2. 检查是否应该修复
        if not repair_plan.get("should_repair", False):
            reason = repair_plan.get("reason", "Agent 判断不应执行修复")
            return False, config_path, {
                "status": "not_repaired",
                "reason": reason,
                "error": reason,
            }

        # 3. 检查修复次数限制
        if self.repair_count >= self.max_repair_attempts:
            return False, config_path, {
                "error": "达到最大修复尝试次数",
                "max_attempts": self.max_repair_attempts,
            }

        self.repair_count += 1

        # 4. 加载当前配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            return False, config_path, {"error": f"无法加载配置: {e}"}

        # 5. 执行配置变更
        strategy = repair_plan.get("strategy")
        config_changes = repair_plan.get("config_changes", {})
        comment_out_fields = repair_plan.get("comment_out_fields", [])

        changes_made = []
        backup_path = None

        try:
            # 应用配置变更
            for key, value in config_changes.items():
                old_value = config.get(key, "<missing>")
                config[key] = value
                changes_made.append(f"{key}: {old_value} -> {value}")

            # 处理需要注释掉的字段（通过特殊标记）
            if comment_out_fields:
                config["_commented_out_by_repair"] = comment_out_fields
                changes_made.append(f"注释字段: {comment_out_fields}")

            # 6. 备份原配置
            backup_path = self._backup_config(config_path)

            # 7. 保存修复后的配置
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            # 8. 记录修复日志
            repair_details = {
                "attempt": self.repair_count,
                "strategy": strategy,
                "changes": changes_made,
                "backup_path": backup_path,
                "agent_reasoning": repair_plan.get("reasoning", ""),
                "confidence": repair_plan.get("confidence", "unknown"),
            }
            self._log_repair_with_context(config_path, error_context, repair_details)

            return True, config_path, repair_details

        except Exception as e:
            # 如果保存失败，尝试恢复备份
            if backup_path and os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, config_path)
                except Exception:
                    pass

            return False, config_path, {
                "error": f"执行修复时出错: {str(e)}",
                "changes_attempted": changes_made,
            }

    def _backup_config(self, config_path: str) -> str:
        """备份配置文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{config_path}.backup.{timestamp}"
        shutil.copy2(config_path, backup_path)
        return backup_path

    def _log_repair_with_context(
        self,
        config_path: str,
        error_context: Dict[str, Any],
        repair_details: Dict[str, Any]
    ):
        """记录修复日志（包含上下文）"""
        error_signals = error_context.get("error_detection", {})
        signals = error_signals.get("signals", [])

        log_entry = f"""
{'='*60}
修复时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
配置文件: {config_path}
修复次数: {repair_details.get('attempt', 0)}

错误信号:
"""
        for signal in signals:
            log_entry += f"  - 类型: {signal['error_type']}\n"
            log_entry += f"    匹配: {signal['matched_text']}\n"

        log_entry += f"""
修复策略: {repair_details.get('strategy', 'unknown')}
Agent 推理: {repair_details.get('agent_reasoning', 'N/A')}
置信度: {repair_details.get('confidence', 'unknown')}

修改内容:
"""
        for change in repair_details.get("changes", []):
            log_entry += f"  - {change}\n"

        log_entry += f"备份路径: {repair_details.get('backup_path', 'N/A')}\n"
        log_entry += f"{'='*60}\n"

        with open(self.repair_log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)

    def monitor_training(
        self,
        log_file: str,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """监控训练日志，检测错误 - Agent 驱动模式

        Args:
            log_file: 日志文件路径
            timeout: 超时时间（秒）

        Returns:
            监控结果字典，包含错误上下文供 Agent 分析
        """
        result = {
            "status": "running",
            "error_detected": False,
            "error_context": None,
            "log_content": "",
        }

        start_time = time.time()
        last_position = 0

        while time.time() - start_time < timeout:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_position)
                    new_content = f.read()
                    last_position = f.tell()

                    if new_content:
                        result["log_content"] += new_content

                        # 检测错误信号
                        error_signals = self.detect_error_signals(new_content)

                        if error_signals["signals_detected"]:
                            result["status"] = "error_detected"
                            result["error_detected"] = True
                            result["error_context"] = error_signals
                            return result

                        # 检查训练是否成功启动
                        if self._is_training_started(new_content):
                            result["status"] = "success"
                            return result

            time.sleep(5)

        result["status"] = "timeout"
        return result

    def _is_training_started(self, log_content: str) -> bool:
        """检查训练是否已开始"""
        success_patterns = [
            re.compile(r"loss:\s*\d+\.?\d*"),
            re.compile(r"Step:\s*\d+"),
            re.compile(r"global_step\s*[=:]\s*\d+"),
            re.compile(r"Starting\s+training", re.IGNORECASE),
            re.compile(r"Begin\s+training", re.IGNORECASE),
        ]

        for pattern in success_patterns:
            if pattern.search(log_content):
                return True

        return False

    def should_attempt_repair(self, error_context: Dict[str, Any]) -> bool:
        """判断是否应该尝试修复 - 提供给 Agent 的辅助方法

        根据错误信号和当前状态，建议是否值得尝试修复。
        最终决策权在 Agent。

        Args:
            error_context: 错误上下文

        Returns:
            是否建议尝试修复
        """
        # 检查修复次数
        if self.repair_count >= self.max_repair_attempts:
            return False

        # 检查错误信号
        error_signals = error_context.get("error_detection", {})
        signals = error_signals.get("signals", [])

        if not signals:
            return False

        # 检查是否有不可修复的错误类型
        unrepairable_types = {
            "runtime_error",
            "operator_not_supported",
            "unknown",
        }

        for signal in signals:
            if signal["error_type"] in unrepairable_types:
                return False

        return True


def main():
    """命令行入口 - Agent 驱动模式"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Error handler for XPU training (Agent-driven)'
    )
    parser.add_argument('--log-file', help='Path to training log file')
    parser.add_argument('--config-file', help='Path to config file')
    parser.add_argument('--model-name', help='Model name')
    parser.add_argument(
        '--get-context',
        action='store_true',
        help='Print error context for Agent analysis'
    )

    args = parser.parse_args()

    handler = ErrorHandler()

    if args.get_context and args.log_file:
        # 读取日志
        with open(args.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()

        # 获取错误上下文
        context = handler.get_error_context(
            log_content=log_content,
            config_path=args.config_file,
            model_name=args.model_name
        )

        print("# 错误处理上下文")
        print(yaml.dump(context, default_flow_style=False, allow_unicode=True))

    else:
        print("Agent 驱动模式：请使用 --get-context 获取错误上下文")
        print("示例：python error_handler.py --log-file train.log --config-file config.yaml --get-context")

    return 0


if __name__ == '__main__':
    exit(main())
