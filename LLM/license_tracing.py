import os
import json
import re
from typing import Dict, List, Tuple, Optional
from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch
from difflib import SequenceMatcher
from datetime import datetime

class LlamaLicenseTracer:
    def __init__(self, cache_dir="/root/autodl-tmp/cache"):
        """
        初始化Llama3-8B许可证溯源分析器
        """
        # 设置环境变量
        os.environ['MODELSCOPE_CACHE'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        
        # 检查CUDA可用性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        print(f"使用缓存目录: {cache_dir}")
        print("正在加载Llama3-8B模型...")
        
        try:
            self.model_name = "LLM-Research/Meta-Llama-3-8B-Instruct"
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # 设置pad token（修复attention mask问题）
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                # 或者使用不同的token作为pad token
                # self.tokenizer.pad_token = "<|pad|>"
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 确保模型在正确的设备上
            if not hasattr(self.model, 'device') or self.model.device != self.device:
                self.model = self.model.to(self.device)
            
            print("模型加载成功！")
            print(f"模型设备: {next(self.model.parameters()).device}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
        
        # 常见的开源许可证列表
        self.known_licenses = [
            "MIT", "Apache-2.0", "GPL-2.0", "GPL-3.0", "LGPL-2.1", "LGPL-3.0",
            "BSD-2-Clause", "BSD-3-Clause", "MPL-2.0", "ISC", "Unlicense",
            "CC0-1.0", "AGPL-3.0", "EPL-1.0", "EPL-2.0", "CDDL-1.0", "CDDL-1.1",
            "CPL-1.0", "IPL-1.0", "NPL-1.1", "OSL-3.0", "PHP-3.0", "PostgreSQL",
            "Python-2.0", "QPL-1.0", "Ruby", "W3C", "X11", "Zlib", "Artistic-2.0",
            "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-NC-4.0", "EUPL-1.1", "EUPL-1.2"
        ]
    
    def create_tracing_prompt(self, license_text: str) -> str:
        """
        创建许可证溯源分析的提示词
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a legal expert specializing in software license analysis and tracing. Your task is to identify the parent or base license that a custom license is derived from.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Please analyze the following license text and identify its parent/base license. Many custom licenses are based on well-known open source licenses with additional clauses.

Known licenses to consider: {', '.join(self.known_licenses)}

License text to analyze:
{license_text}

Please provide your analysis in the following JSON format:
{{
    "parent_license": "identified parent license name or 'Unknown'",
    "confidence": "confidence score from 0.0 to 1.0",
    "reasoning": "explanation of why you identified this parent license",
    "key_similarities": ["list of key text similarities or clauses that match the parent license"],
    "additional_clauses": ["list of additional clauses not found in the parent license"],
    "is_custom": "true if this appears to be a custom license, false if it's a standard license"
}}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def analyze_license(self, license_text: str) -> Dict:
        """
        分析单个许可证文本，识别其父许可证
        """
        prompt = self.create_tracing_prompt(license_text)
        
        try:
            # 编码输入并确保在正确的设备上
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4000,
                padding=True,  # 启用padding
                return_attention_mask=True  # 明确返回attention mask
            )
            
            # 将所有输入张量移动到模型所在的设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            print(f"输入设备: {inputs['input_ids'].device}")
            print(f"模型设备: {next(self.model.parameters()).device}")
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],  # 明确传递attention mask
                    max_new_tokens=1000,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # 解码回复
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取助手回复部分
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            else:
                assistant_response = response.split(prompt)[-1].strip()
            
            # 尝试解析JSON回复
            try:
                # 查找JSON部分
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', assistant_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                else:
                    # 如果没有找到JSON，创建默认结果
                    result = self._create_fallback_result(assistant_response)
            except json.JSONDecodeError:
                result = self._create_fallback_result(assistant_response)
            
            # 添加文本相似度分析
            similarity_result = self._calculate_similarity(license_text)
            result['similarity_analysis'] = similarity_result
            
            return result
            
        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            return {
                "parent_license": "Unknown",
                "confidence": 0.0,
                "reasoning": f"分析失败: {str(e)}",
                "key_similarities": [],
                "additional_clauses": [],
                "is_custom": "unknown",
                "error": str(e)
            }
    
    def _create_fallback_result(self, response_text: str) -> Dict:
        """
        当JSON解析失败时创建备用结果
        """
        # 尝试从文本中提取许可证名称
        parent_license = "Unknown"
        for license_name in self.known_licenses:
            if license_name.lower() in response_text.lower():
                parent_license = license_name
                break
        
        return {
            "parent_license": parent_license,
            "confidence": 0.5,
            "reasoning": "基于文本分析的推测结果",
            "key_similarities": [],
            "additional_clauses": [],
            "is_custom": "unknown",
            "raw_response": response_text
        }
    
    def _calculate_similarity(self, license_text: str) -> Dict:
        """
        计算与已知许可证的文本相似度
        """
        license_keywords = {
            "MIT": ["permission", "copyright", "notice", "software", "free of charge"],
            "Apache-2.0": ["apache", "license", "version 2.0", "contributor", "patent"],
            "GPL-3.0": ["gnu", "general public license", "copyleft", "derivative", "source code"],
            "BSD-3-Clause": ["redistribution", "binary", "source", "neither the name", "endorsement"]
        }
        
        similarities = {}
        license_lower = license_text.lower()
        
        for license_name, keywords in license_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in license_lower)
            similarity = matches / len(keywords)
            similarities[license_name] = similarity
        
        # 找到最高相似度
        best_match = max(similarities.items(), key=lambda x: x[1]) if similarities else ("Unknown", 0.0)
        
        return {
            "best_match": best_match[0],
            "best_similarity": best_match[1],
            "all_similarities": similarities
        }
    
    def batch_analyze(self, license_files: List[str], output_dir: str = "tracing_results") -> Dict:
        """
        批量分析多个许可证文件
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = {}
        
        for i, file_path in enumerate(license_files):
            print(f"正在分析 {i+1}/{len(license_files)}: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    license_text = f.read()
                
                result = self.analyze_license(license_text)
                
                # 保存单个结果
                filename = os.path.basename(file_path)
                result_file = os.path.join(output_dir, f"tracing_{filename}.json")
                
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                results[file_path] = result
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                results[file_path] = {"error": str(e)}
        
        # 保存汇总结果
        summary_file = os.path.join(output_dir, "tracing_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results

def main():
    """
    主函数示例
    """
    # 初始化分析器
    tracer = LlamaLicenseTracer(cache_dir="/root/autodl-tmp/cache")
    file_path = '/root/autodl-tmp/red_hadit_license.txt'
    # 示例：分析单个许可证
    sample_license = """
    MIT License with Additional Restrictions
    
    Copyright (c) 2024 Example Company
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    ADDITIONAL RESTRICTION: This software may not be used for military purposes.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print("分析示例许可证...")
    result = tracer.analyze_license(content)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()