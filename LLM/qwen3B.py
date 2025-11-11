import pandas as pd
import json
import os
from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import warnings
import os

# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ["http_proxy"] = "http://127.0.0.1:7897"
# os.environ["https_proxy"] = "http://127.0.0.1:7897"

warnings.filterwarnings('ignore')


class QwenLicensePredictor:
    def __init__(self, model_name="qwen/Qwen2.5-3B-Instruct"):
        """
        初始化Qwen模型预测器
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载模型和tokenizer
        print("正在加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                                       cache_dir='/root/autodl-tmp')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir='/root/autodl-tmp'
        )

        # 许可证条款定义
        self.license_terms = [
            "Distribute", "Modify", "Commercial Use", "Hold Liable", "Include Copyright",
            "Include License", "Sublicense", "Use Trademark", "Private Use", "Disclose Source",
            "State Changes", "Place Warranty", "Include Notice", "Include Original", "Give Credit",
            "Use Patent Claims", "Rename", "Relicense", "Contact Author", "Include Install Instructions",
            "Compensate for Damages", "Statically Link", "Pay Above Use Threshold"
        ]

        # 标签映射
        self.label_mapping = {
            "允许": 1, "可以": 1, "permitted": 1, "allowed": 1,
            "禁止": 2, "不允许": 2, "forbidden": 2, "prohibited": 2,
            "必须": 3, "要求": 3, "required": 3, "mandatory": 3,
            "不适用": 0, "无关": 0, "not applicable": 0, "n/a": 0
        }

    def create_prompt(self, license_text, term):
        """
        创建用于预测的提示词
        """
        prompt = f"""请分析以下许可证文本，判断对于"{term}"这个条款的态度。

许可证文本：
{license_text}

请根据许可证内容，判断对于"{term}"的态度，并只返回以下数字之一：
0 - 不适用/无关
1 - 允许/可以
2 - 禁止/不允许  
3 - 必须/要求

只返回数字，不要其他解释："""
        return prompt

    def predict_single_term(self, license_text, term):
        """
        预测单个条款
        """
        prompt = self.create_prompt(license_text, term)

        try:
            # 生成回复
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # 提取数字
            response = response.strip()
            for char in response:
                if char.isdigit() and char in ['0', '1', '2', '3']:
                    return int(char)

            # 如果没有找到数字，尝试文本匹配
            response_lower = response.lower()
            for text, label in self.label_mapping.items():
                if text in response_lower:
                    return label

            return 0  # 默认返回0

        except Exception as e:
            print(f"预测条款 {term} 时出错: {e}")
            return 0

    def predict_license(self, license_text, license_name):
        """
        预测单个许可证的所有条款
        """
        print(f"正在预测许可证: {license_name}")
        results = {"license_name": license_name}

        for term in tqdm(self.license_terms, desc=f"预测 {license_name}"):
            prediction = self.predict_single_term(license_text, term)
            results[term] = float(prediction)

        return results

    def load_license_texts(self, license_dir):
        """
        加载许可证文本文件
        """
        license_texts = {}

        if not os.path.exists(license_dir):
            print(f"许可证目录不存在: {license_dir}")
            return license_texts

        for filename in os.listdir(license_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(license_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    license_texts[filename] = content
                except Exception as e:
                    print(f"读取文件 {filename} 时出错: {e}")

        return license_texts

    def batch_predict(self, license_dir, output_file):
        """
        批量预测许可证
        """
        # 加载许可证文本
        license_texts = self.load_license_texts(license_dir)

        if not license_texts:
            print("没有找到许可证文件")
            return

        print(f"找到 {len(license_texts)} 个许可证文件")

        # 预测所有许可证
        all_results = []

        for filename, content in license_texts.items():
            result = self.predict_license(content, filename)
            all_results.append(result)

        # 保存结果
        self.save_results(all_results, output_file)
        print(f"预测结果已保存到: {output_file}")

    def save_results(self, results, output_file):
        """
        保存预测结果为CSV格式
        """
        # 创建DataFrame
        df_data = []
        for result in results:
            row = [result["license_name"]]
            for term in self.license_terms:
                row.append(result.get(term, 0.0))
            row.append(0.0)  # 添加最后一列空值
            df_data.append(row)

        # 创建列名
        columns = [""] + self.license_terms + [""]

        # 创建DataFrame并保存
        df = pd.DataFrame(df_data, columns=columns)
        df.to_csv(output_file, index=False)


def main():
    # 配置路径
    license_dir = "/root/autodl-tmp/license_text"  # 许可证文本目录
    output_file = "/root/autodl-tmp/predict_qwen7b.csv"

    # 创建预测器
    predictor = QwenLicensePredictor()

    # 批量预测
    predictor.batch_predict(license_dir, output_file)


if __name__ == "__main__":
    main()