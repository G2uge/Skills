#!/usr/bin/env python3
"""
GPU YAML 配置检索器（Agent 驱动版）

此脚本为 Agent 提供 PaddleFormers 仓库定位和 GPU YAML 检索的基础能力。
复杂的决策逻辑由 Agent 根据 SKILL.md 中的规则自主实现。

核心设计原则：
1. Skill 提供数据和能力，Agent 完成决策
2. 分层搜索能力拆分为独立方法，Agent 按需调用
3. 提供丰富的上下文信息，支持 Agent 语义分析

主要功能：
1. 分层搜索 PaddleFormers 仓库（三层策略）
2. 扫描和提取 YAML 文件信息
3. 提供搜索上下文和候选数据供 Agent 分析
"""

import os
import re
import glob
import sys
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple


class GPUYamlFinder:
    """GPU YAML 配置检索器 - Agent 驱动版"""

    def __init__(self, paddleformers_root: str = None, auto_search: bool = False):
        """
        初始化检索器

        Args:
            paddleformers_root: PaddleFormers 仓库根目录
                              - 如果提供，直接使用该路径
                              - 如果为 None，需要 Agent 显式调用搜索方法
            auto_search: 是否在初始化时自动搜索（默认 False，Agent 驱动）
                        - False: Agent 驱动模式，由 Agent 决定何时/如何搜索
                        - True: 兼容模式，自动执行完整搜索
        """
        self.search_report = []  # 记录搜索过程
        self.search_context = {  # 收集搜索上下文供 Agent 分析
            "layer1_env": {},
            "layer2_workspace": {},
            "layer3_extended": {},
            "environment_info": {},
        }
        self.paddleformers_root = paddleformers_root

        if paddleformers_root:
            # 验证提供的路径
            if self._is_valid_paddleformers(paddleformers_root):
                self.search_report.append(f"✓ 使用提供的 PaddleFormers 路径: {paddleformers_root}")
            else:
                raise FileNotFoundError(
                    f"提供的路径不是有效的 PaddleFormers 仓库: {paddleformers_root}\n"
                    f"有效指示: setup.py, pyproject.toml, paddleformers/, configs/, examples/"
                )
        elif auto_search:
            # 兼容模式：自动搜索（旧行为）
            self.paddleformers_root = self._execute_full_search()
        else:
            # Agent 驱动模式：不自动搜索，等待 Agent 决策
            self.search_report.append("Agent 驱动模式: 等待显式搜索调用")
            self.paddleformers_root = None

    # ============================================================================
    # Layer 1: 环境感知层 - Python 运行环境定位
    # ============================================================================

    def search_layer1_environment(self) -> Dict[str, Any]:
        """
        Layer 1: 环境感知层搜索
        
        基于当前 Python 运行环境定位 PaddleFormers，包括：
        - 已安装的 paddleformers 模块路径
        - site-packages 目录
        - PYTHONPATH 环境变量
        - 虚拟环境相关路径
        
        Returns:
            包含搜索结果和上下文信息的字典
        """
        self.search_report.append("\n[Layer 1] 环境感知层搜索开始...")
        results = {
            "layer": 1,
            "name": "环境感知层",
            "description": "基于 Python 运行环境定位",
            "found_paths": [],
            "search_details": [],
            "environment_info": {},
        }

        # 1. 尝试导入 paddleformers 模块
        try:
            import paddleformers
            module_path = os.path.dirname(paddleformers.__file__)
            package_root = os.path.dirname(module_path)

            results["environment_info"]["module_installed"] = True
            results["environment_info"]["module_path"] = module_path
            results["environment_info"]["package_root"] = package_root

            # 检查路径有效性
            if self._is_valid_paddleformers(package_root):
                results["found_paths"].append({
                    "path": os.path.abspath(package_root),
                    "source": "paddleformers_module",
                    "description": "从 paddleformers 模块解析的安装路径",
                    "priority": 1,
                })
                results["search_details"].append(f"✓ 从模块定位: {package_root}")
            elif self._is_valid_paddleformers(module_path):
                results["found_paths"].append({
                    "path": os.path.abspath(module_path),
                    "source": "paddleformers_module_direct",
                    "description": "从 paddleformers 模块直接路径定位（可能是 editable install）",
                    "priority": 1,
                })
                results["search_details"].append(f"✓ 从模块直接路径定位: {module_path}")
            else:
                results["search_details"].append(f"- 模块路径无效: {package_root}")
        except ImportError:
            results["environment_info"]["module_installed"] = False
            results["search_details"].append("- paddleformers 模块未安装")
        except Exception as e:
            results["search_details"].append(f"- 导入 paddleformers 失败: {e}")

        # 2. 检查 site-packages 目录
        try:
            import site
            site_packages = site.getsitepackages()
            results["environment_info"]["site_packages"] = site_packages

            for sp in site_packages:
                # 检查直接安装
                pf_path = os.path.join(sp, 'paddleformers')
                if os.path.isdir(pf_path) and self._is_valid_paddleformers(pf_path):
                    results["found_paths"].append({
                        "path": os.path.abspath(pf_path),
                        "source": "site_packages",
                        "description": f"site-packages 目录: {sp}",
                        "priority": 2,
                    })
                    results["search_details"].append(f"✓ 从 site-packages 定位: {pf_path}")

                # 检查 editable install (.pth 文件)
                pth_file = os.path.join(sp, 'paddleformers.pth')
                if os.path.exists(pth_file):
                    with open(pth_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                if os.path.isdir(line) and self._is_valid_paddleformers(line):
                                    results["found_paths"].append({
                                        "path": os.path.abspath(line),
                                        "source": "pth_file",
                                        "description": f"editable install 指向: {pth_file}",
                                        "priority": 2,
                                    })
                                    results["search_details"].append(f"✓ 从 .pth 文件定位: {line}")
        except Exception as e:
            results["search_details"].append(f"- 检查 site-packages 失败: {e}")

        # 3. 检查用户 site-packages
        try:
            import site
            user_site = site.getusersitepackages()
            results["environment_info"]["user_site_packages"] = user_site

            if user_site:
                pf_path = os.path.join(user_site, 'paddleformers')
                if os.path.isdir(pf_path) and self._is_valid_paddleformers(pf_path):
                    results["found_paths"].append({
                        "path": os.path.abspath(pf_path),
                        "source": "user_site_packages",
                        "description": f"用户 site-packages: {user_site}",
                        "priority": 3,
                    })
                    results["search_details"].append(f"✓ 从用户 site-packages 定位: {pf_path}")
        except Exception as e:
            results["search_details"].append(f"- 检查用户 site-packages 失败: {e}")

        # 4. 检查 PYTHONPATH
        pythonpath = os.environ.get('PYTHONPATH', '')
        results["environment_info"]["PYTHONPATH"] = pythonpath

        if pythonpath:
            for path in pythonpath.split(os.pathsep):
                path = path.strip()
                if path:
                    if self._is_valid_paddleformers(path):
                        results["found_paths"].append({
                            "path": os.path.abspath(path),
                            "source": "PYTHONPATH",
                            "description": f"PYTHONPATH 环境变量",
                            "priority": 3,
                        })
                        results["search_details"].append(f"✓ 从 PYTHONPATH 定位: {path}")

                    pf_path = os.path.join(path, 'paddleformers')
                    if os.path.isdir(pf_path) and self._is_valid_paddleformers(pf_path):
                        results["found_paths"].append({
                            "path": os.path.abspath(pf_path),
                            "source": "PYTHONPATH_subdirectory",
                            "description": f"PYTHONPATH 子目录",
                            "priority": 3,
                        })
                        results["search_details"].append(f"✓ 从 PYTHONPATH 子目录定位: {pf_path}")

        # 5. 检查 Python 解释器相关路径
        python_exec = sys.executable
        results["environment_info"]["python_executable"] = python_exec

        if python_exec:
            python_dir = os.path.dirname(python_exec)
            venv_root = os.path.dirname(python_dir)
            results["environment_info"]["venv_root"] = venv_root

            possible_paths = [
                os.path.join(venv_root, 'PaddleFormers'),
                os.path.join(venv_root, 'src', 'PaddleFormers'),
                os.path.join(venv_root, 'workspace', 'PaddleFormers'),
                os.path.join(venv_root, 'PaddleFormers-develop'),
                os.path.join(venv_root, 'PaddleFormers-main'),
            ]

            for path in possible_paths:
                if os.path.isdir(path) and self._is_valid_paddleformers(path):
                    results["found_paths"].append({
                        "path": os.path.abspath(path),
                        "source": "venv_directory",
                        "description": f"虚拟环境根目录: {venv_root}",
                        "priority": 4,
                    })
                    results["search_details"].append(f"✓ 从虚拟环境定位: {path}")

        self.search_context["layer1_env"] = results
        self.search_report.extend(results["search_details"])
        return results

    # ============================================================================
    # Layer 2: 工作空间层 - 当前项目目录搜索
    # ============================================================================

    def search_layer2_workspace(self) -> Dict[str, Any]:
        """
        Layer 2: 工作空间层搜索
        
        在当前工作空间和常见开发目录中搜索：
        - 当前工作目录及其子目录
        - 用户主目录下的常见开发路径
        - 环境变量指定的路径
        - 固定的常用服务器路径
        
        Returns:
            包含搜索结果和上下文信息的字典
        """
        self.search_report.append("\n[Layer 2] 工作空间层搜索开始...")
        results = {
            "layer": 2,
            "name": "工作空间层",
            "description": "当前工作空间和常见开发目录",
            "found_paths": [],
            "search_details": [],
            "workspace_info": {},
        }

        # 1. 当前工作目录
        cwd = os.getcwd()
        results["workspace_info"]["current_directory"] = cwd
        results["search_details"].append(f"当前工作目录: {cwd}")

        # 检查当前目录本身
        if self._is_valid_paddleformers(cwd):
            results["found_paths"].append({
                "path": os.path.abspath(cwd),
                "source": "current_directory",
                "description": "当前工作目录就是 PaddleFormers 仓库",
                "priority": 1,
            })
            results["search_details"].append("✓ 当前目录就是 PaddleFormers 仓库")

        # 检查当前目录下的子目录
        subdirs = ['PaddleFormers', 'paddleformers', 'Paddle', 'workspace', 'src', 'projects']
        for subdir in subdirs:
            path = os.path.join(cwd, subdir)
            if os.path.isdir(path) and self._is_valid_paddleformers(path):
                results["found_paths"].append({
                    "path": os.path.abspath(path),
                    "source": "cwd_subdirectory",
                    "description": f"当前目录子目录: {subdir}",
                    "priority": 2,
                })
                results["search_details"].append(f"✓ 在当前目录子目录找到: {subdir}")

        # 向上搜索父目录（最多5层）
        current = cwd
        for i in range(5):
            parent = os.path.dirname(current)
            if parent == current:
                break
            for subdir in ['PaddleFormers', 'paddleformers']:
                path = os.path.join(parent, subdir)
                if os.path.isdir(path) and self._is_valid_paddleformers(path):
                    results["found_paths"].append({
                        "path": os.path.abspath(path),
                        "source": "parent_directory",
                        "description": f"父目录({i+1}层): {parent}",
                        "priority": 3,
                    })
                    results["search_details"].append(f"✓ 在父目录({i+1}层)找到: {path}")
            current = parent

        # 2. 用户主目录下的常见开发目录
        home = os.path.expanduser('~')
        results["workspace_info"]["home_directory"] = home

        home_dev_paths = [
            os.path.join(home, 'PaddleFormers'),
            os.path.join(home, 'workspace', 'PaddleFormers'),
            os.path.join(home, 'projects', 'PaddleFormers'),
            os.path.join(home, 'src', 'PaddleFormers'),
            os.path.join(home, 'code', 'PaddleFormers'),
            os.path.join(home, 'paddlejob', 'PaddleFormers'),
            os.path.join(home, 'paddlejob', 'Gruge', 'PaddleFormers'),
            os.path.join(home, 'work', 'PaddleFormers'),
            os.path.join(home, 'dev', 'PaddleFormers'),
        ]
        
        # 动态扫描 paddlejob 下的子目录（用于发现用户个人工作空间）
        paddlejob_base = os.path.join(home, 'paddlejob')
        if os.path.isdir(paddlejob_base):
            try:
                for subdir in os.listdir(paddlejob_base):
                    subdir_path = os.path.join(paddlejob_base, subdir)
                    if os.path.isdir(subdir_path):
                        # 检查子目录本身
                        if self._is_valid_paddleformers(subdir_path):
                            home_dev_paths.append(subdir_path)
                        # 检查子目录下的 PaddleFormers
                        pf_path = os.path.join(subdir_path, 'PaddleFormers')
                        if os.path.isdir(pf_path):
                            home_dev_paths.append(pf_path)
                        # 检查孙目录（如 zhangxiao_dev/qwen_env 结构）
                        try:
                            for subsubdir in os.listdir(subdir_path):
                                subsubdir_path = os.path.join(subdir_path, subsubdir)
                                if os.path.isdir(subsubdir_path):
                                    pf_path2 = os.path.join(subsubdir_path, 'PaddleFormers')
                                    if os.path.isdir(pf_path2):
                                        home_dev_paths.append(pf_path2)
                        except (PermissionError, OSError):
                            pass
            except (PermissionError, OSError):
                pass

        for path in home_dev_paths:
            if os.path.isdir(path) and self._is_valid_paddleformers(path):
                results["found_paths"].append({
                    "path": os.path.abspath(path),
                    "source": "home_directory",
                    "description": f"用户主目录: {os.path.dirname(path)}",
                    "priority": 4,
                })
                results["search_details"].append(f"✓ 在用户主目录找到: {path}")

        # 3. 环境变量指定的路径
        env_vars = {
            'PADDLEFORMERS_ROOT': os.environ.get('PADDLEFORMERS_ROOT'),
            'PADDLE_HOME': os.environ.get('PADDLE_HOME'),
            'WORKSPACE': os.environ.get('WORKSPACE'),
            'PROJECT_ROOT': os.environ.get('PROJECT_ROOT'),
        }
        results["workspace_info"]["environment_variables"] = env_vars

        for var_name, var_path in env_vars.items():
            if var_path:
                if self._is_valid_paddleformers(var_path):
                    results["found_paths"].append({
                        "path": os.path.abspath(var_path),
                        "source": "environment_variable",
                        "description": f"环境变量 {var_name}",
                        "priority": 2,
                    })
                    results["search_details"].append(f"✓ 从环境变量 {var_name} 定位: {var_path}")

                pf_path = os.path.join(var_path, 'PaddleFormers')
                if os.path.isdir(pf_path) and self._is_valid_paddleformers(pf_path):
                    results["found_paths"].append({
                        "path": os.path.abspath(pf_path),
                        "source": "environment_variable_subdirectory",
                        "description": f"环境变量 {var_name} 子目录",
                        "priority": 3,
                    })
                    results["search_details"].append(f"✓ 从环境变量 {var_name} 子目录定位: {pf_path}")

        # 4. 固定的常用路径
        fixed_paths = [
            "/root/paddlejob/Gruge/PaddleFormers",
            "/root/paddlejob/PaddleFormers",
            "/workspace/PaddleFormers",
            "/root/PaddleFormers",
            "/home/paddle/PaddleFormers",
            "/opt/PaddleFormers",
            "/data/PaddleFormers",
            "/paddle/PaddleFormers",
            "/mnt/PaddleFormers",
        ]
        results["workspace_info"]["fixed_paths_checked"] = fixed_paths

        for path in fixed_paths:
            if os.path.isdir(path) and self._is_valid_paddleformers(path):
                results["found_paths"].append({
                    "path": os.path.abspath(path),
                    "source": "fixed_common_path",
                    "description": "常用服务器固定路径",
                    "priority": 5,
                })
                results["search_details"].append(f"✓ 从固定路径找到: {path}")

        self.search_context["layer2_workspace"] = results
        self.search_report.extend(results["search_details"])
        return results

    # ============================================================================
    # Layer 3: 扩展搜索层 - 更大范围搜索
    # ============================================================================

    def search_layer3_extended(self, search_roots: List[str] = None, max_depth: int = 3) -> Dict[str, Any]:
        """
        Layer 3: 扩展搜索层
        
        在更大范围内搜索 PaddleFormers 仓库：
        - 扫描指定根目录（默认多个常见根目录）
        - 限制搜索深度以控制性能
        - 使用目录特征匹配
        
        Args:
            search_roots: 要扫描的根目录列表，默认为常见系统路径
            max_depth: 最大搜索深度（默认 3）
        
        Returns:
            包含搜索结果和上下文信息的字典
        """
        self.search_report.append(f"\n[Layer 3] 扩展搜索层开始 (深度限制: {max_depth})...")
        results = {
            "layer": 3,
            "name": "扩展搜索层",
            "description": f"更大范围搜索（深度限制: {max_depth}）",
            "found_paths": [],
            "search_details": [],
            "search_parameters": {
                "max_depth": max_depth,
                "search_roots": search_roots or "默认系统路径",
            },
        }

        if search_roots is None:
            search_roots = [
                os.path.expanduser('~'),
                '/workspace',
                '/data',
                '/opt',
                '/root',
                '/home',
                '/paddle',
                '/mnt',
            ]

        for root in search_roots:
            if not os.path.isdir(root):
                continue

            self.search_report.append(f"  扫描: {root}")

            try:
                for dirpath, dirnames, filenames in os.walk(root):
                    # 限制深度
                    depth = dirpath.count(os.sep) - root.count(os.sep)
                    if depth > max_depth:
                        del dirnames[:]
                        continue

                    # 检查目录名是否匹配 PaddleFormers 特征
                    dirname = os.path.basename(dirpath)
                    if 'paddleformers' in dirname.lower() or dirname.startswith('Paddle'):
                        if self._is_valid_paddleformers(dirpath):
                            results["found_paths"].append({
                                "path": os.path.abspath(dirpath),
                                "source": "extended_scan",
                                "description": f"扩展扫描: {root} (深度 {depth})",
                                "priority": depth,
                            })
                            results["search_details"].append(f"✓ 全局搜索找到: {dirpath}")
                            # 找到一个就停止在当前 root 的扫描
                            break

            except PermissionError:
                results["search_details"].append(f"- 无权限访问: {root}")
                continue
            except Exception as e:
                results["search_details"].append(f"- 搜索 {root} 出错: {e}")
                continue

        # 尝试使用系统 find 命令（Linux/Mac）
        try:
            result = subprocess.run(
                ['find', os.path.expanduser('~'), '-maxdepth', '4', '-type', 'd', '-name', '*addleFormers*'],
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line and self._is_valid_paddleformers(line):
                        # 检查是否已存在
                        existing_paths = [p["path"] for p in results["found_paths"]]
                        if os.path.abspath(line) not in existing_paths:
                            results["found_paths"].append({
                                "path": os.path.abspath(line),
                                "source": "system_find_command",
                                "description": "系统 find 命令",
                                "priority": 5,
                            })
                            results["search_details"].append(f"✓ 通过 find 命令找到: {line}")
        except Exception:
            pass

        self.search_context["layer3_extended"] = results
        self.search_report.extend(results["search_details"])
        return results

    # ============================================================================
    # Agent 决策支持方法
    # ============================================================================

    def get_search_context(self) -> Dict[str, Any]:
        """
        获取完整的搜索上下文信息，供 Agent 决策分析
        
        Returns:
            包含所有层级搜索结果和环境信息的完整上下文
        """
        return {
            "search_status": {
                "has_paddleformers_root": self.paddleformers_root is not None,
                "current_root": self.paddleformers_root,
            },
            "environment_summary": {
                "python_executable": sys.executable,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "current_working_directory": os.getcwd(),
                "user_home": os.path.expanduser('~'),
                "platform": sys.platform,
            },
            "layer_results": {
                "layer1_environment": self.search_context.get("layer1_env", {}),
                "layer2_workspace": self.search_context.get("layer2_workspace", {}),
                "layer3_extended": self.search_context.get("layer3_extended", {}),
            },
            "all_found_paths": self._collect_all_found_paths(),
            "search_report": self.get_search_report(),
        }

    def _collect_all_found_paths(self) -> List[Dict[str, Any]]:
        """收集所有层级找到的路径"""
        all_paths = []
        seen_paths = set()

        for layer_key in ["layer1_env", "layer2_workspace", "layer3_extended"]:
            layer_result = self.search_context.get(layer_key, {})
            for path_info in layer_result.get("found_paths", []):
                path = path_info["path"]
                if path not in seen_paths:
                    seen_paths.add(path)
                    all_paths.append(path_info)

        # 按优先级排序
        all_paths.sort(key=lambda x: x.get("priority", 99))
        return all_paths

    def execute_layered_search(self, stop_on_first: bool = True) -> Optional[str]:
        """
        执行完整的分层搜索（Agent 可调用的便捷方法）
        
        Args:
            stop_on_first: 找到第一个有效路径时是否停止（默认 True）
        
        Returns:
            找到的有效路径，或 None
        """
        # Layer 1
        layer1 = self.search_layer1_environment()
        if stop_on_first and layer1["found_paths"]:
            self.paddleformers_root = layer1["found_paths"][0]["path"]
            self.search_report.append(f"\n✓ Layer 1 成功: {self.paddleformers_root}")
            return self.paddleformers_root

        # Layer 2
        layer2 = self.search_layer2_workspace()
        if stop_on_first and layer2["found_paths"]:
            self.paddleformers_root = layer2["found_paths"][0]["path"]
            self.search_report.append(f"\n✓ Layer 2 成功: {self.paddleformers_root}")
            return self.paddleformers_root

        # Layer 3
        layer3 = self.search_layer3_extended()
        if layer3["found_paths"]:
            self.paddleformers_root = layer3["found_paths"][0]["path"]
            self.search_report.append(f"\n✓ Layer 3 成功: {self.paddleformers_root}")
            return self.paddleformers_root

        self.search_report.append("\n✗ 所有层级搜索均未找到 PaddleFormers 仓库")
        return None

    def set_paddleformers_root(self, path: str) -> bool:
        """
        Agent 手动设置 PaddleFormers 根目录
        
        Args:
            path: 指定的路径
        
        Returns:
            是否设置成功
        """
        if self._is_valid_paddleformers(path):
            self.paddleformers_root = os.path.abspath(path)
            self.search_report.append(f"Agent 手动设置路径: {self.paddleformers_root}")
            return True
        else:
            self.search_report.append(f"手动设置路径无效: {path}")
            return False

    # ============================================================================
    # 内部方法
    # ============================================================================

    def _execute_full_search(self) -> Optional[str]:
        """内部方法：执行完整搜索（兼容旧行为）"""
        result = self.execute_layered_search(stop_on_first=True)
        if result:
            return result
        else:
            error_msg = "无法找到 PaddleFormers 仓库\n\n搜索过程:\n" + "\n".join(self.search_report)
            raise FileNotFoundError(error_msg)

    def _is_valid_paddleformers(self, path: str, require_configs: bool = False) -> bool:
        """
        验证是否是有效的 PaddleFormers 仓库
        
        Args:
            path: 要验证的路径
            require_configs: 是否要求包含 configs 目录（用于 YAML 检索场景）
        
        Returns:
            是否是有效的 PaddleFormers 仓库
        """
        if not os.path.isdir(path):
            return False
        
        # 基础指示器（至少需要一个）
        basic_indicators = ["setup.py", "pyproject.toml", "paddleformers", "examples"]
        has_basic = any(os.path.exists(os.path.join(path, ind)) for ind in basic_indicators)
        
        if not has_basic:
            return False
        
        # 如果需要 configs 目录（用于 YAML 检索），额外检查
        if require_configs:
            configs_path = os.path.join(path, "configs")
            examples_path = os.path.join(path, "examples")
            # configs 或 examples 目录应该存在，用于存放 YAML 配置
            return os.path.isdir(configs_path) or os.path.isdir(examples_path)
        
        return True

    def get_search_report(self) -> str:
        """获取搜索过程报告"""
        return "\n".join(self.search_report)

    # ============================================================================
    # GPU YAML 检索方法（保持原有功能）
    # ============================================================================

    def find_candidate_yamls(self, model_name: str) -> Dict[str, Any]:
        """
        查找候选 GPU YAML 文件（Agent 驱动版本）
        
        不再使用固定的评分机制，而是提供原始数据和上下文信息，
        由 Agent 基于语义理解自主选择最合适的模板。

        Args:
            model_name: 模型名称（如 Qwen3-VL-30B-A3B-Thinking）

        Returns:
            包含候选 YAML 信息、模型特征、搜索上下文的数据结构
        """
        # 确保有 paddleformers_root
        if not self.paddleformers_root:
            self.search_report.append("\n错误: 未设置 PaddleFormers 仓库路径")
            self.search_report.append("请先调用 execute_layered_search() 或 set_paddleformers_root()")
            return {
                "model_name": model_name,
                "error": "PaddleFormers 仓库未定位",
                "features": self.extract_model_features(model_name),
                "candidates": [],
                "paddleformers_root": None,
                "search_report": self.get_search_report(),
            }

        # 1. 提取模型特征
        features = self.extract_model_features(model_name)
        self.search_report.append(f"\n查找 GPU YAML: {model_name}")
        self.search_report.append(f"  模型特征: {features}")

        # 2. 扫描候选 YAML 文件
        model_family = features.get("family", model_name.split("-")[0].lower())
        yaml_files = self.scan_yaml_files(model_family)

        if not yaml_files:
            # 尝试更广泛的搜索（不带 family 限制）
            yaml_files = self._scan_all_yaml_files()

        if not yaml_files:
            self.search_report.append("  ✗ 未找到任何候选 YAML 文件")
            return {
                "model_name": model_name,
                "features": features,
                "candidates": [],
                "paddleformers_root": self.paddleformers_root,
                "search_report": self.get_search_report()
            }

        self.search_report.append(f"  扫描到 {len(yaml_files)} 个候选文件")

        # 3. 提取每个候选文件的详细信息（供 Agent 分析）
        candidates = []
        for yaml_path in yaml_files:
            yaml_info = self.extract_yaml_info(yaml_path)
            # 添加额外的元信息，帮助 Agent 理解
            yaml_info["file_path"] = yaml_path
            yaml_info["file_name"] = os.path.basename(yaml_path)
            yaml_info["relative_path"] = os.path.relpath(yaml_path, self.paddleformers_root)
            
            # 提取文件路径中的语义信息（如目录结构暗示的模型类型）
            path_parts = yaml_info["relative_path"].lower().split(os.sep)
            yaml_info["path_hints"] = {
                "model_family_from_path": self._extract_model_family_from_path(path_parts),
                "task_from_path": self._extract_task_from_path(path_parts),
                "device_from_path": self._extract_device_from_path(path_parts)
            }
            
            candidates.append(yaml_info)

        return {
            "model_name": model_name,
            "features": features,
            "candidates": candidates,
            "paddleformers_root": self.paddleformers_root,
            "search_report": self.get_search_report()
        }

    def scan_yaml_files(self, model_family: str) -> List[str]:
        """
        扫描指定模型家族的所有 YAML 配置文件

        Args:
            model_family: 模型家族名称（如 qwen3_vl, llama 等）

        Returns:
            YAML 文件路径列表
        """
        search_patterns = [
            f"{self.paddleformers_root}/{model_family}/configs/*.yaml",
            f"{self.paddleformers_root}/configs/{model_family}/*.yaml",
            f"{self.paddleformers_root}/examples/{model_family}/configs/*.yaml",
            f"{self.paddleformers_root}/examples/{model_family}/*.yaml",
        ]

        all_files = []
        for pattern in search_patterns:
            all_files.extend(glob.glob(pattern, recursive=True))

        return list(set(all_files))

    def _scan_all_yaml_files(self) -> List[str]:
        """扫描 PaddleFormers 仓库中的所有 YAML 配置文件"""
        search_patterns = [
            f"{self.paddleformers_root}/**/configs/*.yaml",
            f"{self.paddleformers_root}/**/examples/**/*.yaml",
            f"{self.paddleformers_root}/*_vl/configs/*.yaml",
            f"{self.paddleformers_root}/*_vl/*.yaml",
            f"{self.paddleformers_root}/**/tests/**/*.yaml",
            f"{self.paddleformers_root}/**/benchmark/**/*.yaml",
        ]

        all_files = []
        for pattern in search_patterns:
            all_files.extend(glob.glob(pattern, recursive=True))

        # 过滤掉非 GPU 配置（包含 xpu 字样的可能是 XPU 配置）
        gpu_files = [f for f in all_files if "xpu" not in os.path.basename(f).lower()]

        return list(set(gpu_files))

    def extract_yaml_info(self, yaml_path: str) -> Dict[str, Any]:
        """
        从 YAML 文件中提取基础信息

        Args:
            yaml_path: YAML 文件路径

        Returns:
            包含基础信息的字典
        """
        info = {
            "yaml_path": yaml_path,
            "filename": os.path.basename(yaml_path),
            "exists": os.path.exists(yaml_path),
            "model_name": "",
            "model_path": "",
            "stage": "",
            "device": "",
            "params": {},
        }

        if not info["exists"]:
            return info

        try:
            import yaml
            with open(yaml_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)

            if content:
                info["model_name"] = content.get("model_name_or_path", "")
                info["model_path"] = content.get("model_name_or_path", "")
                info["stage"] = content.get("stage", "")
                info["device"] = content.get("device", "")
                info["template"] = content.get("template", "")
                info["params"] = {
                    "per_device_train_batch_size": content.get("per_device_train_batch_size", 1),
                    "gradient_accumulation_steps": content.get("gradient_accumulation_steps", 1),
                    "learning_rate": content.get("learning_rate", ""),
                    "max_seq_len": content.get("max_seq_len", ""),
                    "num_train_epochs": content.get("num_train_epochs", 1),
                }
        except Exception as e:
            info["error"] = str(e)

        return info

    def extract_model_features(self, model_name: str) -> Dict[str, Any]:
        """
        从模型名称中提取特征信息
        
        Args:
            model_name: 模型名称（如 Qwen3-VL-30B-A3B-Thinking）

        Returns:
            模型特征字典
        """
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

        # 提取系列
        family_patterns = [
            (r"qwen3", "qwen3"),
            (r"qwen2\.5", "qwen2.5"),
            (r"qwen2", "qwen2"),
            (r"qwen", "qwen"),
            (r"llama", "llama"),
            (r"deepseek", "deepseek"),
            (r"chatglm", "chatglm"),
            (r"baichuan", "baichuan"),
            (r"internlm", "internlm"),
        ]
        for pattern, family in family_patterns:
            if re.search(pattern, name_lower):
                features["family"] = family
                break

        # 提取参数规模
        size_match = re.search(r'(\d+)(b|B)', model_name)
        if size_match:
            features["size"] = size_match.group(0)
            features["size_value"] = int(size_match.group(1))

        # 提取激活参数（MoE 模型）
        active_match = re.search(r'a(\d+)(b|B)', name_lower)
        if active_match:
            features["active_size"] = active_match.group(0)
            features["active_size_value"] = int(active_match.group(1))
            features["structure"].append("moe")

        # 提取结构特征
        if "vl" in name_lower or "vision" in name_lower:
            features["structure"].append("vl")
            features["variant"] = "vl"
        elif "text" in name_lower:
            features["structure"].append("text")
            features["variant"] = "text"

        # 提取任务类型
        task_patterns = [
            (r'thinking', 'thinking'),
            (r'instruct', 'instruct'),
            (r'chat', 'chat'),
            (r'sft', 'sft'),
            (r'pretrain', 'pretrain'),
            (r'rlhf', 'rlhf'),
            (r'dpo', 'dpo'),
        ]
        for pattern, task in task_patterns:
            if re.search(pattern, name_lower):
                features["task_type"] = task
                break

        return features

    def _extract_model_family_from_path(self, path_parts: List[str]) -> List[str]:
        """从路径中提取模型家族信息"""
        hints = []
        family_keywords = ["qwen", "llama", "deepseek", "glm", "ernie", "baichuan", "chatglm"]
        for part in path_parts:
            for keyword in family_keywords:
                if keyword in part.lower():
                    hints.append(part)
                    break
        return hints

    def _extract_task_from_path(self, path_parts: List[str]) -> List[str]:
        """从路径中提取任务类型信息"""
        hints = []
        task_keywords = ["sft", "pretrain", "dpo", "rlhf", "instruct", "chat", "lora"]
        for part in path_parts:
            for keyword in task_keywords:
                if keyword in part.lower():
                    hints.append(part)
                    break
        return hints

    def _extract_device_from_path(self, path_parts: List[str]) -> str:
        """从路径中提取设备类型信息"""
        for part in path_parts:
            if "xpu" in part.lower():
                return "xpu"
            elif "gpu" in part.lower():
                return "gpu"
            elif "iluvatar" in part.lower():
                return "iluvatar"
        return "unknown"


def main():
    """命令行入口 - 用于测试和调试"""
    import argparse

    parser = argparse.ArgumentParser(description='GPU YAML 检索器 (Agent 驱动版)')
    parser.add_argument('model_name', nargs='?', help='模型名称（如 Qwen3-VL-30B）')
    parser.add_argument('--paddleformers-root', help='直接指定 PaddleFormers 仓库根目录')
    parser.add_argument('--scan', action='store_true', help='扫描并列出所有候选 YAML')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细的搜索过程')
    parser.add_argument('--test-search', action='store_true', help='测试分层搜索功能')

    args = parser.parse_args()

    # 初始化
    if args.paddleformers_root:
        finder = GPUYamlFinder(paddleformers_root=args.paddleformers_root)
    else:
        # Agent 驱动模式
        finder = GPUYamlFinder(auto_search=False)

    print("=" * 70)
    print("PaddleFormers GPU YAML 检索器 (Agent 驱动版)")
    print("=" * 70)

    # 测试分层搜索
    if args.test_search or not args.paddleformers_root:
        print("\n📋 执行分层搜索...")
        
        # Layer 1
        print("\n" + "-" * 50)
        print("Layer 1: 环境感知层")
        print("-" * 50)
        layer1 = finder.search_layer1_environment()
        print(f"找到路径数: {len(layer1['found_paths'])}")
        for p in layer1['found_paths']:
            print(f"  ✓ {p['path']}")
            print(f"    来源: {p['source']}, 优先级: {p['priority']}")
        
        # Layer 2
        print("\n" + "-" * 50)
        print("Layer 2: 工作空间层")
        print("-" * 50)
        layer2 = finder.search_layer2_workspace()
        print(f"找到路径数: {len(layer2['found_paths'])}")
        for p in layer2['found_paths']:
            print(f"  ✓ {p['path']}")
            print(f"    来源: {p['source']}, 优先级: {p['priority']}")
        
        # 如果还没找到，执行 Layer 3
        if not layer1['found_paths'] and not layer2['found_paths']:
            print("\n" + "-" * 50)
            print("Layer 3: 扩展搜索层")
            print("-" * 50)
            layer3 = finder.search_layer3_extended()
            print(f"找到路径数: {len(layer3['found_paths'])}")
            for p in layer3['found_paths']:
                print(f"  ✓ {p['path']}")
                print(f"    来源: {p['source']}, 优先级: {p['priority']}")
        
        # 执行完整搜索
        print("\n" + "=" * 70)
        print("执行完整分层搜索...")
        result = finder.execute_layered_search(stop_on_first=True)
        
        if result:
            print(f"\n✓ 最终选择路径: {result}")
        else:
            print("\n✗ 未找到 PaddleFormers 仓库")
            print("\n完整搜索报告:")
            print(finder.get_search_report())
            return 1

    print(f"\nPaddleFormers root: {finder.paddleformers_root}")
    
    if args.verbose:
        print("\n搜索过程报告:")
        print(finder.get_search_report())

    # 如果提供了模型名称，继续查找 YAML
    if args.model_name:
        print()
        print("=" * 70)
        print(f"查找模型: {args.model_name}")
        print("=" * 70)

        # 提取模型特征
        features = finder.extract_model_features(args.model_name)
        print("\n模型特征解析:")
        for key, value in features.items():
            print(f"  {key}: {value}")
        print()

        # 获取候选 YAML
        candidate_data = finder.find_candidate_yamls(args.model_name)
        
        if candidate_data.get("error"):
            print(f"错误: {candidate_data['error']}")
            return 1
        
        print(f"扫描到 {len(candidate_data['candidates'])} 个 YAML 文件:")
        
        for i, candidate in enumerate(candidate_data['candidates'], 1):
            print(f"\n  [{i}] {candidate['file_name']}")
            print(f"      路径: {candidate['relative_path']}")
            if candidate.get('stage'):
                print(f"      Stage: {candidate['stage']}")
            if candidate.get('device'):
                print(f"      Device: {candidate['device']}")
            if candidate.get('path_hints'):
                hints = candidate['path_hints']
                print(f"      路径语义:")
                print(f"        - 模型系列: {hints.get('model_family_from_path', [])}")
                print(f"        - 任务类型: {hints.get('task_from_path', [])}")
                print(f"        - 设备类型: {hints.get('device_from_path', 'unknown')}")

        print("\n" + "=" * 70)
        print("注意: YAML 选择由 Agent 基于语义理解自主决策")
        print("Skill 仅提供候选数据和上下文信息")
        print("=" * 70)

    return 0


if __name__ == '__main__':
    exit(main())
