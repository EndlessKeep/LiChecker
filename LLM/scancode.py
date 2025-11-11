import subprocess
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

class ScanCodeLicenseScanner:
    """
    使用ScanCode工具扫描自定义许可证并识别其父许可证
    """
    
    def __init__(self):
        self.scancode_cmd = "scancode"
        
    def check_scancode_installation(self) -> bool:
        """
        检查ScanCode是否已安装
        """
        try:
            result = subprocess.run([self.scancode_cmd, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def scan_license_file(self, license_file_path: str, output_format: str = "json") -> Dict:
        """
        扫描单个许可证文件
        
        Args:
            license_file_path: 许可证文件路径
            output_format: 输出格式 (json, jsonlines, csv, html, spdx)
            
        Returns:
            扫描结果字典
        """
        if not os.path.exists(license_file_path):
            raise FileNotFoundError(f"许可证文件不存在: {license_file_path}")
            
        if not self.check_scancode_installation():
            raise RuntimeError("ScanCode未安装或无法访问，请先安装: pip install scancode-toolkit")
        
        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{output_format}', delete=False) as temp_file:
            temp_output = temp_file.name
        
        try:
            # 构建ScanCode命令 - 修复版本
            cmd = [
                self.scancode_cmd,
                "--license",  # 扫描许可证
                "--license-text",  # 包含许可证文本
                "--classify",  # 分类选项（license-clarity-score的前置要求）
                "--license-clarity-score",  # 许可证清晰度评分
                "--timeout", "300",  # 超时设置
                f"--{output_format}", temp_output,  # 输出格式和文件
                license_file_path
            ]
            
            print(f"正在扫描许可证文件: {license_file_path}")
            print(f"执行命令: {' '.join(cmd)}")
            
            # 执行ScanCode命令
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                raise RuntimeError(f"ScanCode执行失败: {result.stderr}")
            
            # 读取扫描结果
            if output_format == "json":
                with open(temp_output, 'r', encoding='utf-8') as f:
                    scan_result = json.load(f)
            else:
                with open(temp_output, 'r', encoding='utf-8') as f:
                    scan_result = {"raw_output": f.read()}
            
            return scan_result
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    def extract_license_info(self, scan_result: Dict) -> Dict:
        """
        从扫描结果中提取许可证信息
        
        Args:
            scan_result: ScanCode扫描结果
            
        Returns:
            提取的许可证信息
        """
        license_info = {
            "detected_licenses": [],
            "license_expressions": [],
            "parent_licenses": [],
            "primary_license": None,  # 新增：主要许可证
            "license_clarity_score": None,
            "file_info": {}
        }
        
        # 用于去重的集合
        seen_licenses = set()
        
        # 首先尝试从新版本的license_detections中提取
        if "license_detections" in scan_result:
            for detection in scan_result["license_detections"]:
                license_expression = detection.get("license_expression")
                license_expression_spdx = detection.get("license_expression_spdx")
                
                if license_expression_spdx and license_expression_spdx not in seen_licenses:
                    seen_licenses.add(license_expression_spdx)
                    
                    license_data = {
                        "key": license_expression,
                        "name": license_expression_spdx,
                        "category": "Permissive",
                        "owner": None,
                        "homepage_url": None,
                        "text_url": None,
                        "spdx_license_key": license_expression_spdx,
                        "score": 100,
                        "matched_text": None
                    }
                    
                    license_info["detected_licenses"].append(license_data)
                    license_info["license_expressions"].append(license_expression_spdx)
                    
                    # 设置主要许可证（第一个非专有许可证）
                    if not license_info["primary_license"] and not license_expression_spdx.startswith("LicenseRef-"):
                        license_info["primary_license"] = license_expression_spdx
        
        # 兼容旧版本格式：从files中提取
        if "files" in scan_result:
            for file_data in scan_result["files"]:
                if file_data.get("type") == "file":
                    license_info["file_info"] = {
                        "name": file_data.get("name"),
                        "path": file_data.get("path"),
                        "size": file_data.get("size")
                    }
                    
                    # 提取许可证检测结果
                    if "licenses" in file_data:
                        for license_data in file_data["licenses"]:
                            spdx_key = license_data.get("spdx_license_key")
                            if spdx_key and spdx_key not in seen_licenses:
                                seen_licenses.add(spdx_key)
                                
                                license_info["detected_licenses"].append({
                                    "key": license_data.get("key"),
                                    "name": license_data.get("name"),
                                    "category": license_data.get("category"),
                                    "owner": license_data.get("owner"),
                                    "homepage_url": license_data.get("homepage_url"),
                                    "text_url": license_data.get("text_url"),
                                    "spdx_license_key": spdx_key,
                                    "score": license_data.get("score"),
                                    "matched_text": license_data.get("matched_text")
                                })
                                
                                # 设置主要许可证
                                if not license_info["primary_license"] and not spdx_key.startswith("LicenseRef-"):
                                    license_info["primary_license"] = spdx_key
                    
                    # 提取许可证表达式
                    if "license_expressions" in file_data:
                        for expr in file_data["license_expressions"]:
                            if expr not in license_info["license_expressions"]:
                                license_info["license_expressions"].append(expr)
                    
                    # 提取许可证清晰度评分
                    if "license_clarity_score" in file_data:
                        license_info["license_clarity_score"] = file_data["license_clarity_score"]
        
        # 从summary中提取许可证清晰度评分（新版本格式）
        if "summary" in scan_result and "license_clarity_score" in scan_result["summary"]:
            license_info["license_clarity_score"] = scan_result["summary"]["license_clarity_score"]
        
        # 识别可能的父许可证
        license_info["parent_licenses"] = self._identify_parent_licenses(
            license_info["detected_licenses"]
        )
        
        # 如果没有设置主要许可证，使用第一个检测到的许可证
        if not license_info["primary_license"] and license_info["detected_licenses"]:
            license_info["primary_license"] = license_info["detected_licenses"][0]["spdx_license_key"]
        
        return license_info
    
    def _identify_parent_licenses(self, detected_licenses: List[Dict]) -> List[str]:
        """
        基于检测到的许可证识别可能的父许可证
        
        Args:
            detected_licenses: 检测到的许可证列表
            
        Returns:
            可能的父许可证列表
        """
        parent_licenses = []
        
        # 常见的许可证族群映射
        license_families = {
            "apache": ["Apache-1.0", "Apache-1.1", "Apache-2.0"],
            "bsd": ["BSD-2-Clause", "BSD-3-Clause", "BSD-4-Clause"],
            "gpl": ["GPL-1.0", "GPL-2.0", "GPL-3.0"],
            "lgpl": ["LGPL-2.0", "LGPL-2.1", "LGPL-3.0"],
            "mit": ["MIT"],
            "mpl": ["MPL-1.0", "MPL-1.1", "MPL-2.0"]
        }
        
        for license_data in detected_licenses:
            license_key = license_data.get("key", "").lower()
            spdx_key = license_data.get("spdx_license_key", "").lower()
            
            # 检查许可证族群
            for family, versions in license_families.items():
                if any(version.lower() in license_key or version.lower() in spdx_key 
                      for version in versions):
                    if family not in parent_licenses:
                        parent_licenses.append(family.upper())
            
            # 直接添加SPDX标识符作为父许可证
            if license_data.get("spdx_license_key"):
                parent_licenses.append(license_data["spdx_license_key"])
        
        return list(set(parent_licenses))  # 去重
    
    def scan_and_analyze(self, license_file_path: str) -> Dict:
        """
        扫描并分析许可证文件的完整流程
        
        Args:
            license_file_path: 许可证文件路径
            
        Returns:
            完整的分析结果
        """
        print(f"开始扫描许可证文件: {license_file_path}")
        
        # 执行扫描
        scan_result = self.scan_license_file(license_file_path)
        
        # 提取许可证信息
        license_info = self.extract_license_info(scan_result)
        
        # 生成分析报告
        analysis_result = {
            "input_file": license_file_path,
            "scan_timestamp": scan_result.get("headers", [{}])[0].get("start_timestamp"),
            "license_analysis": license_info,
            "raw_scan_result": scan_result
        }
        
        return analysis_result
    
    def batch_scan_licenses(self, license_directory: str) -> Dict:
        """
        批量扫描许可证文件夹
        
        Args:
            license_directory: 包含许可证文件的目录
            
        Returns:
            批量扫描结果
        """
        if not os.path.exists(license_directory):
            raise FileNotFoundError(f"目录不存在: {license_directory}")
        
        results = {}
        license_files = []
        
        # 查找许可证文件
        for root, dirs, files in os.walk(license_directory):
            for file in files:
                if any(keyword in file.lower() for keyword in 
                      ['license', 'licence', 'copying', 'copyright']):
                    license_files.append(os.path.join(root, file))
        
        print(f"找到 {len(license_files)} 个许可证文件")
        
        # 逐个扫描
        for license_file in license_files:
            try:
                print(f"\n正在处理: {license_file}")
                result = self.scan_and_analyze(license_file)
                results[license_file] = result
            except Exception as e:
                print(f"扫描失败 {license_file}: {str(e)}")
                results[license_file] = {"error": str(e)}
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """
        保存扫描结果到文件
        
        Args:
            results: 扫描结果
            output_file: 输出文件路径
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")


def main():
    """
    主函数 - 使用示例
    """
    scanner = ScanCodeLicenseScanner()
    
    # 检查ScanCode安装
    if not scanner.check_scancode_installation():
        print("错误: ScanCode未安装，请运行: pip install scancode-toolkit")
        return
    
    # 示例1: 扫描单个许可证文件
    license_file = "D:\PythonProject\LiResolver_copy\license_txt\cherry-studio.txt"  # 替换为实际路径
    
    if os.path.exists(license_file):
        try:
            result = scanner.scan_and_analyze(license_file)
            
            # 打印分析结果
            print("\n=== 许可证分析结果 ===")
            print(f"文件: {result['input_file']}")
            
            license_analysis = result['license_analysis']
            print(f"\n检测到的许可证数量: {len(license_analysis['detected_licenses'])}")
            
            for i, license_data in enumerate(license_analysis['detected_licenses'], 1):
                print(f"\n许可证 {i}:")
                print(f"  名称: {license_data['name']}")
                print(f"  标识符: {license_data['key']}")
                print(f"  SPDX标识符: {license_data['spdx_license_key']}")
                print(f"  类别: {license_data['category']}")
                print(f"  匹配分数: {license_data['score']}")
                print(f"  所有者: {license_data['owner']}")
                print(f"  主页: {license_data['homepage_url']}")
            
            print(f"\n可能的父许可证: {', '.join(license_analysis['parent_licenses'])}")
            print(f"许可证清晰度评分: {license_analysis['license_clarity_score']}")
            
            # 保存详细结果
            scanner.save_results(result, "license_scan_result.json")
            
        except Exception as e:
            print(f"扫描失败: {str(e)}")
    else:
        print(f"许可证文件不存在: {license_file}")
        print("\n请将 'license_file' 变量设置为您的自定义许可证文件路径")
    
    # 示例2: 批量扫描许可证目录
    # license_directory = "path/to/license/directory"
    # if os.path.exists(license_directory):
    #     batch_results = scanner.batch_scan_licenses(license_directory)
    #     scanner.save_results(batch_results, "batch_license_scan_results.json")


if __name__ == "__main__":
    main()