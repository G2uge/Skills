#!/usr/bin/env python3
"""
XPU 配置生成器（Agent 驱动版）

此脚本为 Agent 提供 GPU→XPU 配置转换的基础能力。
复杂的转换决策逻辑由 AI 根据 SKILL.md 中的规则自主实现。
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class XPUConfigGenerator:
    """XPU 配置生成器 - Agent 驱动版"""

    def __init__(self, mapping_rules_path: str = None):
        """初始化配置生成器"""
        if mapping_rules_path is None:
            mapping_rules_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'templates',
                'mapping_rules.yaml'
            )

        self.mapping_rules = self._load_yaml(mapping_rules_path)
        self.reference_config = self._load_reference_config()

    def _load_yaml(self, path: str) -> Dict:
        """加载 YAML 文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_reference_config(self) -> Dict:
        """加载参考配置"""
        ref_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'reference_configs',
            'xpu_reference.yaml'
        )
        return self._load_yaml(ref_path)

    def get_conversion_context(self, gpu_config_path: str, model_name: str) -> Dict[str, Any]:
        """
        获取配置转换的完整上下文信息，供 Agent 执行转换
        
        这是 Agent 驱动模式的核心方法，提供所有必要的信息和规则，
        由 Agent 基于 SKILL.md 指导完成实际的配置转换。
        """
        gpu_config = self._load_yaml(gpu_config_path)
        model_features = self._extract_model_features(model_name)
        applicable_model_rules = self._get_applicable_model_rules(model_name)

        context = {
            "source": {
                "gpu_config_path": gpu_config_path,
                "gpu_config": gpu_config,
                "model_name": model_name,
                "model_features": model_features,
            },
            "rules": {
                "field_mappings": self.mapping_rules.get('field_mappings', {}),
                "parameter_transformations": self.mapping_rules.get('parameter_transformations', {}),
                "structural_adjustments": self.mapping_rules.get('structural_adjustments', {}),
                "pattern_rules": self.mapping_rules.get('pattern_rules', {}),
                "model_specific": applicable_model_rules,
                "conversion_workflow": self.mapping_rules.get('conversion_workflow', {}),
            },
            "reference": {
                "default_values": self.mapping_rules.get('default_values', {}),
                "reference_config": self.reference_config,
            },
            "validation": {
                "validation_rules": self.mapping_rules.get('validation_rules', {}),
            },
        }

        return context

    def _extract_model_features(self, model_name: str) -> Dict[str, Any]:
        """从模型名称中提取特征信息"""
        features = {
            "original_name": model_name,
            "family": "",
            "variant": "",
            "size": "",
            "size_value": 0,
            "task_type": "",
            "structure": [],
        }

        name_lower = model_name.lower()

        family_patterns = [
            (r"qwen3", "qwen3"),
            (r"qwen2\.5", "qwen2.5"),
            (r"llama", "llama"),
            (r"deepseek", "deepseek"),
            (r"chatglm", "chatglm"),
        ]
        for pattern, family in family_patterns:
            if re.search(pattern, name_lower):
                features["family"] = family
                break

        size_match = re.search(r'(\d+)(b|B)', model_name)
        if size_match:
            features["size"] = size_match.group(0)
            features["size_value"] = int(size_match.group(1))

        active_match = re.search(r'a(\d+)(b|B)', name_lower)
        if active_match:
            features["active_size"] = active_match.group(0)
            features["structure"].append("moe")

        if "vl" in name_lower:
            features["structure"].append("vl")
            features["variant"] = "vl"

        task_patterns = [
            (r'thinking', 'thinking'),
            (r'instruct', 'instruct'),
            (r'sft', 'sft'),
            (r'pretrain', 'pretrain'),
        ]
        for pattern, task in task_patterns:
            if re.search(pattern, name_lower):
                features["task_type"] = task
                break

        return features

    def _get_applicable_model_rules(self, model_name: str) -> Dict[str, Any]:
        """获取适用于当前模型的特定规则"""
        model_rules = self.mapping_rules.get('model_specific_rules', {})
        applicable_rules = {}
        for pattern, rules in model_rules.items():
            if re.search(pattern, model_name):
                applicable_rules[pattern] = rules
        return applicable_rules

    def save_xpu_config(
        self,
        xpu_config: Dict[str, Any],
        model_name: str,
        output_dir: str = None,
        gpu_config_path: str = None,
        generation_report: str = None
    ) -> Tuple[str, str]:
        """保存 Agent 生成的 XPU 配置"""
        if output_dir is None:
            output_dir = './checkpoints'

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_model_name = model_name.replace('-', '_').replace('.', '_')
        config_filename = f'train_{safe_model_name}_xpu_{timestamp}.yaml'

        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, config_filename)

        xpu_config['_meta'] = {
            'generated_at': timestamp,
            'model_name': model_name,
            'source_gpu_config': gpu_config_path,
            'generator_version': '2.0-agent-driven',
        }

        self._save_yaml(xpu_config, config_path)

        generation_method = (
            f"基于Agent驱动的GPU→XPU配置转换:\n"
            f"  - 源GPU配置: {gpu_config_path}\n"
            f"  - 模型: {model_name}\n"
            f"  - 转换方式: Agent根据SKILL.md规则自主执行"
        )

        return config_path, generation_method

    def _save_yaml(self, config: Dict, path: str):
        """保存 YAML 文件"""
        config_to_save = {k: v for k, v in config.items() if not k.startswith('_')}
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description='XPU Config Generator (Agent-driven)')
    parser.add_argument('--gpu-config', help='Path to GPU YAML config')
    parser.add_argument('--model-name', required=True, help='Model name')
    parser.add_argument('--output-dir', default='./checkpoints', help='Output directory')
    parser.add_argument('--get-context', action='store_true', help='Print conversion context')
    
    args = parser.parse_args()
    generator = XPUConfigGenerator()

    if args.get_context and args.gpu_config:
        context = generator.get_conversion_context(args.gpu_config, args.model_name)
        print("# GPU→XPU 配置转换上下文")
        print(yaml.dump(context, default_flow_style=False, allow_unicode=True))
    else:
        print("Agent 驱动模式：请使用 --get-context 获取转换上下文")

    return 0


if __name__ == '__main__':
    exit(main())
