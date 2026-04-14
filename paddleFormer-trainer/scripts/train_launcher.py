#!/usr/bin/env python3
"""
XPU 训练启动器 - Agent驱动版本
整合配置生成、Agent监控和错误处理的完整流程
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# 导入其他模块
sys.path.insert(0, os.path.dirname(__file__))
from gpu_yaml_finder import GPUYamlFinder
from config_generator import XPUConfigGenerator
from error_handler import ErrorHandler, ErrorType, RepairStrategy
from training_monitor_agent import TrainingMonitorAgent, TrainingStatus


class TrainLauncher:
    """训练启动器 - Agent驱动架构"""

    def __init__(
        self,
        paddleformers_root: str = None,
        output_base_dir: str = "./checkpoints"
    ):
        """
        初始化训练启动器

        Args:
            paddleformers_root: PaddleFormers 仓库根目录
            output_base_dir: 输出基础目录
        """
        self.output_base_dir = output_base_dir
        self.finder = GPUYamlFinder(paddleformers_root)
        self.generator = XPUConfigGenerator()
        self.error_handler = ErrorHandler()

        # 确保输出目录存在
        os.makedirs(output_base_dir, exist_ok=True)

    def prepare_training(
        self,
        model_name: str,
        gpu_yaml_path: str = None,
        custom_params: Dict[str, Any] = None,
        allow_fallback: bool = False
    ) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        """
        准备训练配置和启动脚本（Agent 驱动版本）

        Args:
            model_name: 模型名称
            gpu_yaml_path: GPU YAML 配置文件路径（由 Agent 选择后传入，可选）
            custom_params: 自定义参数
            allow_fallback: 是否允许在找不到 GPU YAML 时使用参考配置生成

        Returns:
            (XPU YAML 路径, 启动脚本路径, 配置信息, 候选 YAML 数据)
        """
        print(f"\n{'='*60}")
        print(f"准备训练: {model_name}")
        print(f"{'='*60}\n")

        # 1. 获取候选 YAML 数据（Agent 驱动）
        print("[1/4] 获取候选 GPU 配置...")
        candidate_data = self.finder.find_candidate_yamls(model_name)
        
        # 2. 生成 XPU YAML
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_model_name = model_name.replace('-', '_').replace('.', '_')
        output_name = f"train_{safe_model_name}_xpu_{timestamp}"
        output_dir = os.path.join(self.output_base_dir, output_name)
        os.makedirs(output_dir, exist_ok=True)

        if gpu_yaml_path:
            print(f"  ✓ 找到 GPU 配置: {gpu_yaml_path}")
            print("\n[2/4] 基于 GPU 配置生成 XPU 配置...")

            # 基于 GPU 配置生成（推荐方式）
            xpu_yaml_path, xpu_config, generation_method = self.generator.generate_xpu_config(
                gpu_config_path=gpu_yaml_path,
                model_name=model_name,
                output_dir=output_dir,
                custom_params=custom_params
            )

            print(f"  ✓ 生成 XPU 配置: {xpu_yaml_path}")
            print(f"\n📋 配置生成方式:")
            print(f"   ✅ 基于GPU配置转换（推荐方式）")
            print(f"   - 源GPU配置: {gpu_yaml_path}")
            print(f"   - {generation_method}")

        else:
            # 未找到 GPU YAML
            print(f"  ✗ 未找到模型 {model_name} 的 GPU 配置")

            if not allow_fallback:
                # 强制模式：报错并退出
                error_msg = (
                    f"\n❌ 错误：未找到 {model_name} 的 GPU YAML 配置\n\n"
                    f"📋 搜索过程:\n{self.finder.get_search_report()}\n\n"
                    f"💡 建议操作:\n"
                    f"   1. 确认模型名称拼写正确\n"
                    f"   2. 检查 PaddleFormers 仓库路径: {self.finder.paddleformers_root}\n"
                    f"   3. 手动指定GPU配置路径\n"
                    f"   4. 如需使用参考配置生成，请设置 allow_fallback=True"
                )
                raise FileNotFoundError(error_msg)

            # 回退模式：使用参考配置
            print("\n⚠️  [2/4] 使用参考配置生成 XPU 配置（回退方案）...")

            xpu_yaml_path, xpu_config, generation_method = self.generator.generate_xpu_config_from_reference(
                model_name=model_name,
                output_dir=output_dir,
                custom_params=custom_params
            )

            print(f"  ✓ 生成 XPU 配置: {xpu_yaml_path}")
            print(f"\n📋 配置生成方式:")
            print(f"   ⚠️ 基于参考配置生成（回退方案）")
            print(f"   - 未找到模型 {model_name}的GPU YAML配置")
            print(f"   - 使用 reference_configs/xpu_reference.yaml 作为基础")
            print(f"   - ⚠️ 建议人工复核关键参数后再启动训练")

        # 3. 生成启动脚本
        print("\n[3/4] 生成启动脚本...")
        script_path = self._generate_launch_script(
            model_name=model_name,
            config_path=xpu_yaml_path,
            output_dir=output_dir,
            num_xpus=8,
            xpu_devices="0,1,2,3,4,5,6,7"
        )

        print(f"  ✓ 生成启动脚本: {script_path}")

        # 4. 汇总配置信息
        print("\n[4/4] 汇总配置信息...")
        config_info = {
            "model_name": model_name,
            "gpu_yaml_path": gpu_yaml_path,
            "xpu_yaml_path": xpu_yaml_path,
            "script_path": script_path,
            "output_dir": output_dir,
            "logging_dir": xpu_config.get("logging_dir", "./vdl_log"),
            "training_params": {
                "num_train_epochs": xpu_config.get("num_train_epochs", 1),
                "max_steps": xpu_config.get("max_steps", 100),
                "batch_size": xpu_config.get("per_device_train_batch_size", 1),
                "learning_rate": xpu_config.get("learning_rate", "1.0e-4"),
            },
            "xpu_params": {
                "device": xpu_config.get("device", "xpu"),
                "bkcl_timeout": xpu_config.get("bkcl_timeout", 1000),
                "num_xpus": 8,
            }
        }

        return xpu_yaml_path, script_path, config_info, candidate_data

    def _generate_launch_script(
        self,
        model_name: str,
        config_path: str,
        output_dir: str,
        num_xpus: int = 8,
        xpu_devices: str = "0,1,2,3,4,5,6,7"
    ) -> str:
        """
        生成启动脚本

        Args:
            model_name: 模型名称
            config_path: XPU YAML 配置文件路径
            output_dir: 输出目录
            num_xpus: XPU 设备数量
            xpu_devices: XPU 设备列表

        Returns:
            启动脚本路径
        """
        # 读取模板
        template_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'templates',
            'xpu_train.sh.template'
        )

        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        # 替换变量
        script_content = template.replace('{{GENERATED_TIME}}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        script_content = script_content.replace('{{MODEL_NAME}}', model_name)
        script_content = script_content.replace('{{CONFIG_FILE}}', config_path)
        script_content = script_content.replace('{{OUTPUT_DIR}}', output_dir)
        script_content = script_content.replace('{{NUM_XPUS}}', str(num_xpus))
        script_content = script_content.replace('{{XPU_DEVICES}}', xpu_devices)

        # 保存脚本
        safe_model_name = model_name.replace('-', '_').replace('.', '_')
        script_filename = f"run_train_{safe_model_name}_xpu.sh"
        script_path = os.path.join(output_dir, script_filename)

        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # 添加执行权限
        os.chmod(script_path, 0o755)

        return script_path

    def launch_training(
        self,
        script_path: str,
        use_agent_monitor: bool = True,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        启动训练任务 - Agent驱动版本

        Args:
            script_path: 启动脚本路径
            use_agent_monitor: 是否使用Agent监控
            timeout: 超时时间

        Returns:
            启动结果字典
        """
        print(f"\n{'='*60}")
        print(f"启动训练")
        print(f"{'='*60}\n")

        result = {
            "success": False,
            "pid": None,
            "log_file": None,
            "error": None,
            "agent_result": None,
        }

        # 执行启动脚本
        try:
            # 切换到脚本所在目录
            script_dir = os.path.dirname(script_path)

            process = subprocess.Popen(
                ['bash', script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=script_dir
            )

            # 等待脚本完成启动
            time.sleep(2)

            # 查找 PID 文件
            output_dir = script_dir
            pid_file = os.path.join(output_dir, '.train_pid')

            # 等待 PID 文件创建
            for _ in range(10):
                if os.path.exists(pid_file):
                    with open(pid_file, 'r') as f:
                        pid = f.read().strip()
                        if pid:
                            result["pid"] = pid
                            break
                time.sleep(1)

            if result["pid"]:
                result["log_file"] = os.path.join(output_dir, 'train.log')

                print(f"✓ 训练进程已启动")
                print(f"  PID: {result['pid']}")
                print(f"  日志: {result['log_file']}")

                # Agent驱动的训练启动监控
                if use_agent_monitor:
                    print(f"\n🤖 启动Agent监控...")
                    print(f"   模式: Agent驱动训练状态判定")
                    print(f"   目标: 检测首个loss输出作为训练成功标志")
                    
                    # 创建Agent实例
                    agent = TrainingMonitorAgent(timeout=timeout, poll_interval=5)
                    
                    # 执行Agent监控
                    agent_result = agent.monitor_training_start(
                        result["log_file"],
                        callback=lambda msg: print(msg)
                    )
                    
                    result["agent_result"] = agent_result
                    
                    # 根据Agent判定结果设置最终状态
                    if agent_result["success"]:
                        result["success"] = True
                        result["status"] = "running"
                    else:
                        result["success"] = False
                        result["status"] = agent_result.get("status", "failed")
                        result["error"] = agent_result.get("reasoning", "Agent判定训练启动失败")
                else:
                    # 传统模式：仅检查进程是否存在
                    print(f"\n⚠️  使用传统模式（无Agent监控）")
                    result["success"] = True

            else:
                result["error"] = "无法获取训练进程 PID"
                print(f"✗ {result['error']}")

        except Exception as e:
            result["error"] = str(e)
            print(f"✗ 启动失败: {e}")

        return result

    def run_with_repair(
        self,
        model_name: str,
        max_attempts: int = 3,
        custom_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        运行训练并支持自动修复 - Agent驱动版本

        Args:
            model_name: 模型名称
            max_attempts: 最大尝试次数
            custom_params: 自定义参数

        Returns:
            最终运行结果
        """
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            print(f"\n{'='*60}")
            print(f"第 {attempt}/{max_attempts} 次尝试")
            print(f"{'='*60}")

            try:
                # 1. 准备配置
                xpu_yaml_path, script_path, config_info, candidate_data = self.prepare_training(
                    model_name,
                    custom_params=custom_params
                )

                # 2. 启动训练（Agent驱动监控）
                launch_result = self.launch_training(script_path, use_agent_monitor=True)

                if not launch_result["success"]:
                    # 启动失败，检查是否可以修复
                    print(f"\n✗ 启动失败: {launch_result['error']}")
                    
                    # 获取Agent监控结果
                    agent_result = launch_result.get("agent_result", {})
                    
                    # 检查是否是Agent检测到的错误
                    if agent_result and agent_result.get("status") == "error":
                        final_analysis = agent_result.get("final_analysis", {})
                        metrics = final_analysis.get("metrics", {})
                        error_type = metrics.get("error_type", "unknown")
                        
                        print(f"\n⚠️  Agent检测到错误类型: {error_type}")
                        
                        # 尝试自动修复（仅针对特定错误类型）
                        if error_type in ["out_of_memory", "communication_timeout", "config_error"]:
                            print(f"   尝试自动修复...")
                            
                            # 构建修复策略
                            repair_strategy = self._decide_repair_strategy(error_type, xpu_yaml_path)
                            
                            if repair_strategy:
                                repair_success, repaired_path, repair_details = self.error_handler.execute_repair(
                                    xpu_yaml_path,
                                    repair_strategy["strategy"],
                                    repair_strategy["changes"],
                                    repair_strategy.get("comment_out", [])
                                )
                                
                                if repair_success:
                                    print(f"\n✓ 已自动修复配置")
                                    print(f"   修改: {repair_details['changes']}")
                                    continue  # 修复后重试
                                else:
                                    print(f"\n✗ 自动修复失败: {repair_details.get('error')}")
                    
                    if attempt < max_attempts:
                        print("\n准备重试...")
                        continue
                    else:
                        return {
                            "success": False,
                            "error": launch_result["error"],
                            "attempts": attempt,
                            "agent_result": agent_result,
                        }

                # 3. 启动成功（Agent已验证loss输出）
                print(f"\n✅ 训练已成功启动并正常运行！")
                print(f"   Agent判定: 训练状态正常")
                
                agent_result = launch_result.get("agent_result", {})
                if agent_result:
                    print(f"   启动耗时: {agent_result.get('elapsed_time', 0):.1f}秒")
                    final_analysis = agent_result.get("final_analysis", {})
                    metrics = final_analysis.get("metrics", {})
                    if metrics.get("loss_values"):
                        print(f"   检测到的Loss值: {metrics['loss_values'][:3]}")
                
                return {
                    "success": True,
                    "pid": launch_result["pid"],
                    "config_path": xpu_yaml_path,
                    "script_path": script_path,
                    "output_dir": config_info["output_dir"],
                    "attempts": attempt,
                    "agent_result": agent_result,
                }

            except Exception as e:
                print(f"\n✗ 发生异常: {e}")
                import traceback
                traceback.print_exc()

                if attempt < max_attempts:
                    print("准备重试...")
                else:
                    return {
                        "success": False,
                        "error": str(e),
                        "attempts": attempt,
                    }

        return {
            "success": False,
            "error": "达到最大尝试次数",
            "attempts": attempt,
        }
    
    def _decide_repair_strategy(self, error_type: str, config_path: str) -> Optional[Dict[str, Any]]:
        """
        Agent决策：根据错误类型决定修复策略
        
        Args:
            error_type: 错误类型
            config_path: 配置文件路径
            
        Returns:
            修复策略字典
        """
        # 加载当前配置
        config = self.error_handler._load_config(config_path)
        
        if error_type == "out_of_memory":
            # OOM修复策略：减小batch size或增加accumulation
            current_accum = config.get("gradient_accumulation_steps", 8)
            new_accum = min(current_accum * 2, 64)
            
            return {
                "strategy": RepairStrategy.REDUCE_BATCH_SIZE,
                "changes": {"gradient_accumulation_steps": new_accum},
                "comment_out": [],
                "reasoning": f"OOM错误，增加gradient_accumulation_steps {current_accum} -> {new_accum}"
            }
        
        elif error_type == "communication_timeout":
            # 通信超时修复策略：增加timeout
            current_timeout = config.get("bkcl_timeout", 1000)
            new_timeout = min(current_timeout * 2, 5000)
            
            return {
                "strategy": RepairStrategy.INCREASE_TIMEOUT,
                "changes": {"bkcl_timeout": new_timeout},
                "comment_out": [],
                "reasoning": f"通信超时，增加bkcl_timeout {current_timeout} -> {new_timeout}"
            }
        
        elif error_type == "config_error":
            # 配置错误：尝试从参考配置补全
            return {
                "strategy": RepairStrategy.FROM_REFERENCE,
                "changes": {"device": "xpu"},  # 至少确保device字段
                "comment_out": [],
                "reasoning": "配置错误，尝试补全关键字段"
            }
        
        return None

    def print_summary(self, config_info: Dict[str, Any]):
        """打印配置摘要"""
        print(f"\n{'='*60}")
        print("训练配置摘要")
        print(f"{'='*60}")
        print(f"\n模型名称: {config_info['model_name']}")
        print(f"模型路径: {config_info.get('model_path', 'N/A')}")

        print(f"\n配置文件:")
        print(f"  - XPU YAML: {config_info['xpu_yaml_path']}")
        print(f"  - 启动脚本: {config_info['script_path']}")

        print(f"\n训练参数:")
        params = config_info.get('training_params', {})
        print(f"  - 训练阶段: {config_info.get('stage', 'N/A')}")
        print(f"  - Epochs: {params.get('num_train_epochs', 'N/A')}")
        print(f"  - Max Steps: {params.get('max_steps', 'N/A')}")
        print(f"  - Batch Size: {params.get('batch_size', 'N/A')}")
        print(f"  - Learning Rate: {params.get('learning_rate', 'N/A')}")

        print(f"\n输出路径:")
        print(f"  - 输出目录: {config_info['output_dir']}")
        print(f"  - 训练日志: {config_info['logging_dir']}")
        print(f"  - 分布式日志: {config_info['output_dir']}/paddleformers_dist_log/")

        print(f"\nXPU 配置:")
        xpu_params = config_info.get('xpu_params', {})
        print(f"  - 设备: {xpu_params.get('device', 'N/A')}")
        print(f"  - 设备数量: {xpu_params.get('num_xpus', 'N/A')}")
        print(f"  - BKCL Timeout: {xpu_params.get('bkcl_timeout', 'N/A')}")
        print(f"{'='*60}\n")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Launch XPU training with Agent-driven monitoring'
    )
    parser.add_argument('model_name', help='Model name')
    parser.add_argument(
        '--paddleformers-root',
        help='PaddleFormers repository root'
    )
    parser.add_argument(
        '--output-dir',
        default='./checkpoints',
        help='Output directory'
    )
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=3,
        help='Maximum repair attempts'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only prepare configs without launching'
    )
    parser.add_argument(
        '--no-agent',
        action='store_true',
        help='Disable Agent monitoring (use legacy mode)'
    )

    args = parser.parse_args()

    launcher = TrainLauncher(
        paddleformers_root=args.paddleformers_root,
        output_base_dir=args.output_dir
    )

    if args.dry_run:
        # 仅准备配置
        xpu_yaml, script_path, config_info, candidate_data = launcher.prepare_training(args.model_name)
        launcher.print_summary(config_info)
        print(f"\n✓ 配置已生成，但未启动训练")
        print(f"  手动启动命令: bash {script_path}")
    else:
        # 启动训练（支持自动修复）
        if args.no_agent:
            print("⚠️  使用传统模式（无Agent监控）")
            xpu_yaml, script_path, config_info, _ = launcher.prepare_training(args.model_name)
            result = launcher.launch_training(script_path, use_agent_monitor=False)
        else:
            print("🤖 使用Agent驱动模式")
            result = launcher.run_with_repair(
                args.model_name,
                max_attempts=args.max_attempts
            )

        if result["success"]:
            print(f"\n✅ 训练启动成功！")
            print(f"  PID: {result['pid']}")
            print(f"  输出目录: {result['output_dir']}")
            print(f"  监控命令: tail -f {result['output_dir']}/paddleformers_dist_log/workerlog.0")
            
            # 显示Agent监控摘要
            agent_result = result.get("agent_result")
            if agent_result:
                print(f"\n📊 Agent监控摘要:")
                print(f"   启动耗时: {agent_result.get('elapsed_time', 0):.1f}秒")
                print(f"   最终状态: {agent_result.get('status', 'unknown')}")
        else:
            print(f"\n✗ 训练启动失败")
            print(f"  错误: {result['error']}")
            print(f"  尝试次数: {result['attempts']}")
            
            # 显示Agent详细分析
            agent_result = result.get("agent_result")
            if agent_result and agent_result.get("recommendations"):
                print(f"\n💡 Agent建议:")
                for rec in agent_result["recommendations"]:
                    print(f"   - {rec}")
            return 1

    return 0


if __name__ == '__main__':
    exit(main())
