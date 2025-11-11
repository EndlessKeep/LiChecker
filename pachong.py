import requests
import time
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import subprocess

class GitHubTrendingCrawler:
    def __init__(self, token=None):
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-Trending-Crawler"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"
    
    def get_trending_repos(self, since="weekly", language=None):
        """获取热门仓库（通过搜索API模拟）"""
        since_map = {
            "daily": "daily",
            "weekly": "weekly", 
            "monthly": "monthly"
        }
        
        date_since = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        query = f"created:>{date_since} stars:>100"
        
        if language and language != "any":
            query += f" language:{language}"
            
        url = f"{self.base_url}/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 50
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()["items"]
        except requests.exceptions.RequestException as e:
            print(f"获取仓库列表失败: {e}")
            return []
    
    def get_repo_details(self, owner, repo):
        """获取仓库详细信息"""
        url = f"{self.base_url}/repos/{owner}/{repo}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"获取仓库详情失败 {owner}/{repo}: {e}")
            return None

def save_repos_info(repos, filename="github_trending_repos.json"):
    """保存仓库信息到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(repos, f, indent=2, ensure_ascii=False)
    print(f"已保存 {len(repos)} 个仓库信息到 {filename}")

def load_repos_info(filename="github_trending_repos.json"):
    """从文件加载仓库信息"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

class LicenseScanner:
    def __init__(self, output_dir="scan_results"):
        self.output_dir = output_dir
        self.repos_dir = os.path.join(output_dir, "repositories")
        self.results_dir = os.path.join(output_dir, "scan_results")
        
        # 创建目录
        os.makedirs(self.repos_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def clone_repository(self, repo_url, repo_name):
        """克隆仓库到本地"""
        repo_path = os.path.join(self.repos_dir, repo_name)
        
        if os.path.exists(repo_path):
            print(f"仓库 {repo_name} 已存在，跳过克隆")
            return repo_path
        
        try:
            print(f"正在克隆 {repo_name}...")
            subprocess.run([
                "git", "clone", 
                "--depth", "1",  # 只克隆最近的一次提交
                repo_url, 
                repo_path
            ], check=True, capture_output=True)
            return repo_path
        except subprocess.CalledProcessError as e:
            print(f"克隆仓库 {repo_name} 失败: {e}")
            return None
    
    def scan_license_with_scancode(self, repo_path, repo_name):
        """使用scancode扫描许可证"""
        output_file = os.path.join(self.results_dir, f"{repo_name}_scan.json")
        
        try:
            print(f"正在扫描 {repo_name} 的许可证...")
            # 运行scancode扫描
            result = subprocess.run([
                "scancode",
                "--license",  # 扫描许可证
                "--copyright",  # 扫描版权信息
                "--package",   # 扫描包信息
                "--json", output_file,
                repo_path
            ], capture_output=True, text=True, check=True)
            
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"扫描 {repo_name} 失败: {e}")
            return None
    
    def analyze_scan_results(self, scan_result_file):
        """分析扫描结果，提取许可证信息"""
        if not os.path.exists(scan_result_file):
            return None
        
        try:
            with open(scan_result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            licenses = {}
            
            # 提取文件级别的许可证信息
            for file in data.get("files", []):
                file_licenses = file.get("licenses", [])
                for license_info in file_licenses:
                    key = license_info.get("key", "unknown")
                    name = license_info.get("name", "Unknown")
                    spdx_id = license_info.get("spdx_license_key", "")
                    
                    if key not in licenses:
                        licenses[key] = {
                            "name": name,
                            "spdx_id": spdx_id,
                            "files": [],
                            "count": 0
                        }
                    
                    licenses[key]["files"].append(file.get("path", ""))
                    licenses[key]["count"] += 1
            
            # 提取包级别的许可证信息
            for package in data.get("packages", []):
                package_licenses = package.get("declared_license_expression", "")
                if package_licenses:
                    if "custom" not in licenses:
                        licenses["custom"] = {
                            "name": "Custom License",
                            "spdx_id": "",
                            "files": [],
                            "count": 0,
                            "package_licenses": []
                        }
                    licenses["custom"]["package_licenses"].append(package_licenses)
            
            return licenses
        except Exception as e:
            print(f"分析扫描结果失败: {e}")
            return None

    def scan_repositories(self, repos, limit=10):
        """批量扫描仓库"""
        results = []
        
        for i, repo in enumerate(tqdm(repos[:limit])):
            print(f"\n处理仓库 {i+1}/{min(limit, len(repos))}: {repo['name']}")
            
            # 克隆仓库
            repo_path = self.clone_repository(repo['clone_url'], repo['name'])
            if not repo_path:
                continue
            
            # 扫描许可证
            scan_file = self.scan_license_with_scancode(repo_path, repo['name'])
            if not scan_file:
                continue
            
            # 分析结果
            license_info = self.analyze_scan_results(scan_file)
            
            results.append({
                'repo_name': repo['name'],
                'repo_url': repo['html_url'],
                'stars': repo['stargazers_count'],
                'description': repo.get('description', ''),
                'license_info': license_info,
                'scan_file': scan_file
            })
            
            # 添加延迟，避免请求过快
            time.sleep(1)
        
        return results

def main():
    # GitHub token（可选，用于提高API限制）
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # 从环境变量获取
    
    # 初始化爬虫和扫描器
    crawler = GitHubTrendingCrawler(token=GITHUB_TOKEN)
    scanner = LicenseScanner()
    
    print("=== GitHub热门仓库许可证扫描 ===")
    
    # 步骤1: 获取热门仓库
    print("\n1. 获取热门仓库...")
    trending_repos = crawler.get_trending_repos(since="weekly", language="python")
    print(f"找到 {len(trending_repos)} 个热门仓库")
    
    # 保存仓库信息
    save_repos_info(trending_repos)
    
    # 步骤2: 扫描许可证
    print("\n2. 开始扫描许可证...")
    scan_results = scanner.scan_repositories(trending_repos, limit=5)  # 限制数量避免过载
    
    # 步骤3: 生成报告
    print("\n3. 生成扫描报告...")
    generate_report(scan_results)

def generate_report(scan_results):
    """生成扫描报告"""
    report_data = []
    
    for result in scan_results:
        license_summary = "未检测到许可证"
        custom_licenses = []
        
        if result['license_info']:
            licenses = result['license_info']
            license_list = []
            
            for key, info in licenses.items():
                if key == "custom":
                    custom_licenses = info.get('package_licenses', [])
                else:
                    license_list.append(f"{info['name']} ({info['count']} 文件)")
            
            license_summary = "; ".join(license_list) if license_list else "检测到自定义许可证"
        
        report_data.append({
            '仓库名称': result['repo_name'],
            '星标数': result['stars'],
            '仓库URL': result['repo_url'],
            '许可证摘要': license_summary,
            '自定义许可证': ", ".join(custom_licenses) if custom_licenses else "无",
            '描述': result['description'][:100] + '...' if result['description'] and len(result['description']) > 100 else result['description']
        })
    
    # 保存为CSV
    df = pd.DataFrame(report_data)
    df.to_csv('license_scan_report.csv', index=False, encoding='utf-8-sig')
    
    # 保存为JSON
    with open('license_scan_report.json', 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n扫描完成！共处理 {len(scan_results)} 个仓库")
    print("报告已保存至: license_scan_report.csv 和 license_scan_report.json")
    
    # 打印摘要
    print("\n=== 扫描摘要 ===")
    for item in report_data:
        print(f"\n{item['仓库名称']} (⭐{item['星标数']})")
        print(f"许可证: {item['许可证摘要']}")
        if item['自定义许可证'] != "无":
            print(f"自定义许可证: {item['自定义许可证']}")

if __name__ == "__main__":
    main()