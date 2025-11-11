import json
import os
from openai import OpenAI
import warnings

warnings.filterwarnings('ignore')

class DeepSeekLicenseTracer:
    def __init__(self, api_key="your_api_key", base_url="https://www.chataiapi.com/v1"):
        """
        基于DeepSeek的许可证溯源分析器，只分析父许可证
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print("DeepSeek客户端初始化成功！")
    
    def create_parent_license_prompt(self, license_text):
        """
        创建用于识别父许可证的提示词
        """
        prompt = f"""You are a legal expert specializing in software licenses. Please analyze the following license text and identify its parent/base license.

License text:
{license_text[:2000]}...

Based on the license content, what is the parent/base license? Please respond with only the license name (e.g., "GPL-2.0", "MIT", "Apache-2.0", "BSD-3-Clause", etc.).

If you cannot identify a clear parent license, respond with "Unknown".

Parent license:"""
        return prompt
    
    def identify_parent_license(self, license_text):
        """
        使用DeepSeek识别父许可证
        """
        prompt = self.create_parent_license_prompt(license_text)
        
        try:
            completion = self.client.chat.completions.create(
                model="deepseek-r1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            # 获取最终答案
            response = completion.choices[0].message.content.strip()
            
            # 可选：打印思考过程（如果需要调试）
            if hasattr(completion.choices[0].message, 'reasoning_content') and completion.choices[0].message.reasoning_content:
                print(f"DeepSeek思考过程: {completion.choices[0].message.reasoning_content[:200]}...")
            
            return self.extract_license_name(response)
            
        except Exception as e:
            print(f"DeepSeek API调用出错: {e}")
            return "Unknown"
    
    def extract_license_name(self, response):
        """
        从模型响应中提取许可证名称
        """
        # 常见许可证名称模式
        common_licenses = [
            "GPL-2.0", "GPL-3.0", "LGPL-2.1", "LGPL-3.0",
            "MIT", "Apache-2.0", "Apache-1.1", "Apache-1.0",
            "BSD-2-Clause", "BSD-3-Clause", "BSD-4-Clause",
            "MPL-2.0", "MPL-1.1", "EPL-1.0", "EPL-2.0",
            "CDDL-1.0", "CDDL-1.1", "CPL-1.0", "IPL-1.0",
            "ISC", "Zlib", "Unlicense", "CC0-1.0"
        ]
        
        response_upper = response.upper()
        
        # 直接匹配常见许可证
        for license_name in common_licenses:
            if license_name.upper() in response_upper:
                return license_name
        
        # 模糊匹配
        if "GPL" in response_upper:
            if "3" in response or "3.0" in response:
                return "GPL-3.0"
            elif "2" in response or "2.0" in response:
                return "GPL-2.0"
            else:
                return "GPL"
        elif "MIT" in response_upper:
            return "MIT"
        elif "APACHE" in response_upper:
            if "2" in response:
                return "Apache-2.0"
            else:
                return "Apache-1.0"
        elif "BSD" in response_upper:
            if "3" in response:
                return "BSD-3-Clause"
            elif "2" in response:
                return "BSD-2-Clause"
            else:
                return "BSD"
        
        # 如果没有匹配到，返回原始响应的前50个字符
        return response[:50] if response else "Unknown"
    
    def analyze_license_file(self, license_file_path):
        """
        分析单个许可证文件
        """
        try:
            with open(license_file_path, 'r', encoding='utf-8') as f:
                license_text = f.read()
            
            print(f"正在分析许可证文件: {license_file_path}")
            parent_license = self.identify_parent_license(license_text)
            
            result = {
                "license_file": license_file_path,
                "parent_license": parent_license,
                "analysis_status": "success",
                "model_used": "deepseek-r1"
            }
            
            return result
            
        except Exception as e:
            print(f"分析许可证文件 {license_file_path} 时出错: {e}")
            return {
                "license_file": license_file_path,
                "error": str(e),
                "analysis_status": "failed",
                "model_used": "deepseek-r1"
            }
    
    def batch_analyze_folder(self, folder_path, output_file=None):
        """
        批量分析文件夹中的所有文件
        """
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return []
        
        results = []
        
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # 跳过目录
            if os.path.isdir(file_path):
                continue
            
            # 分析文件
            result = self.analyze_license_file(file_path)
            results.append(result)
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n批量分析结果已保存到: {output_file}")
        
        return results

def test_deepseek_license():
    """
    测试DeepSeek许可证分析
    """
    print("=== 测试DeepSeek许可证分析 ===")
    
    try:
        # 创建分析器
        tracer = DeepSeekLicenseTracer()
        
        # 分析单个许可证文件（请根据实际路径修改）
        license_file = "/root/autodl-tmp/cherry/cherry-studio.txt"  # 示例路径
        
        if os.path.exists(license_file):
            print("开始许可证溯源分析...")
            result = tracer.analyze_license_file(license_file)
            
            print("\n分析结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 保存结果
            output_file = "/root/autodl-tmp/deepseek_license_analysis.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\n结果已保存到: {output_file}")
        else:
            print(f"许可证文件不存在: {license_file}")
            print("请修改文件路径或使用batch_analyze_folder方法分析整个文件夹")
        
    except Exception as e:
        print(f"分析失败: {e}")
        result = {
            "error": "DeepSeek许可证分析失败",
            "details": str(e)
        }
        print("\n分析结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

def test_batch_analysis():
    """
    测试批量分析文件夹
    """
    print("=== 测试批量分析 ===")
    
    try:
        # 创建分析器
        tracer = DeepSeekLicenseTracer()
        
        # 批量分析RQ3文件夹（请根据实际路径修改）
        folder_path = "/root/autodl-tmp/RQ3"  # 示例路径
        output_file = "/root/autodl-tmp/batch_deepseek_analysis.json"
        
        if os.path.exists(folder_path):
            print(f"开始批量分析文件夹: {folder_path}")
            results = tracer.batch_analyze_folder(folder_path, output_file)
            
            print(f"\n批量分析完成，共分析了 {len(results)} 个文件")
            
            # 统计结果
            success_count = sum(1 for r in results if r.get('analysis_status') == 'success')
            print(f"成功分析: {success_count} 个文件")
            print(f"分析失败: {len(results) - success_count} 个文件")
            
        else:
            print(f"文件夹不存在: {folder_path}")
            print("请创建RQ3文件夹或修改folder_path变量")
        
    except Exception as e:
        print(f"批量分析失败: {e}")

if __name__ == "__main__":
    # 选择测试模式
    print("选择测试模式:")
    print("1. 单个文件分析")
    print("2. 批量文件夹分析")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        test_deepseek_license()
    elif choice == "2":
        test_batch_analysis()
    else:
        print("无效选择，默认运行单个文件分析")
        test_deepseek_license()