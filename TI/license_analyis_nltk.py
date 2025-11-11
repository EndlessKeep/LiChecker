# 改进的许可证分析接口 - 加入NLTK分句
import torch
import json
import re
import os
import nltk
from transformers import AutoTokenizer, AutoConfig
from typing import List, Dict, Any

# 下载NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 导入原有模块
from EE5.ner_model.model import NERModel  
from EE5.ner_model.config import Config as NERConfig 
from EE5.ner_model.data_utils import load_vocab 
from RE.license_re import load_re_model, predict as predict_re, postprocess_predictions 
from RE.data.dialogue import WIKI80 

class ImprovedLicenseAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_model = None
        self.ner_tokenizer = None
        self.idx_to_tag = None
        self.re_lit_model = None
        self.re_tokenizer = None
        self.re_args = None
        self.id2rel = None
        self.wiki80_processor = None
        
    def load_models(self):
        """加载NER和RE模型"""
        if self.ner_model is not None and self.re_lit_model is not None:
            return
            
        # 加载NER模型
        print("Loading NER model...")
        ner_config = NERConfig() 
        ner_config.dir_model = r'D:\PythonProject\LiResolver_copy\EE5\results\test'
        ner_config.filename_tags = r'D:\PythonProject\LiResolver_copy\EE5\data\tags.txt'

        ner_config.vocab_tags = load_vocab(ner_config.filename_tags)
        ner_config.tag2idx = {tag: idx for idx, tag in enumerate(ner_config.vocab_tags)}
        ner_config.ntags = len(ner_config.vocab_tags)

        self.ner_model = NERModel(ner_config) 
        ner_model_path = fr'{ner_config.dir_model}\model_6_19\\' 
        self.ner_model.restore_session(ner_model_path) 
        self.ner_model.to(self.device)
        self.ner_model.eval()

        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_config.bert_model_name)
        self.idx_to_tag = {idx: tag for tag, idx in ner_config.tag2idx.items()}
        print("NER model loaded.")

        # 加载RE模型
        print("Loading RE model...")
        self.re_args, self.re_lit_model = load_re_model() 
        self.re_tokenizer = self.re_lit_model.tokenizer 
        self.re_lit_model.to(self.device)
        self.re_lit_model.eval()

        with open(r'D:\PythonProject\LiResolver_copy\RE\dataset\ossl2\rel2id.json', 'r') as f:
            rel2id = json.load(f)
        self.id2rel = {v: k for k, v in rel2id.items()}
        print("RE model loaded.")
        
        self.wiki80_processor = WIKI80(self.re_args)

    def split_license_text(self, license_text: str) -> List[str]:
        """使用NLTK进行智能分句"""
        # 预处理：保护重要格式
        text = self._preprocess_text(license_text)
        
        # 使用NLTK分句
        sentences = nltk.sent_tokenize(text)
        
        # 后处理：合并被错误分割的句子
        sentences = self._postprocess_sentences(sentences)
        
        # 过滤相关句子
        relevant_sentences = self._filter_relevant_sentences(sentences)
        
        return relevant_sentences
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本，保护重要格式"""
        # 保护条款编号
        text = re.sub(r'(\d+\.\d*[a-z]*)', r'<CLAUSE_NUM>\1</CLAUSE_NUM>', text)
        
        # 保护法律条款引用
        text = re.sub(r'(Section|Article|Clause)\s+([IVX\d]+)', r'<SECTION_REF>\1 \2</SECTION_REF>', text)
        
        return text
    
    def _postprocess_sentences(self, sentences: List[str]) -> List[str]:
        """后处理句子，合并被错误分割的部分"""
        processed = []
        i = 0
        
        while i < len(sentences):
            current = sentences[i].strip()
            
            # 如果句子以条款编号开始，尝试合并后续句子
            if re.match(r'^\d+\.', current) or re.match(r'^[a-z]\)', current):
                j = i + 1
                while j < len(sentences):
                    next_sent = sentences[j].strip()
                    if not re.match(r'^\d+\.', next_sent) and not re.match(r'^[a-z]\)', next_sent):
                        current += " " + next_sent
                        j += 1
                    else:
                        break
                i = j
            else:
                i += 1
            
            # 恢复保护的格式
            current = self._restore_protected_format(current)
            processed.append(current)
        
        return processed
    
    def _restore_protected_format(self, text: str) -> str:
        """恢复被保护的格式"""
        text = re.sub(r'<CLAUSE_NUM>(.*?)</CLAUSE_NUM>', r'\1', text)
        text = re.sub(r'<SECTION_REF>(.*?)</SECTION_REF>', r'\1', text)
        return text
    
    def _filter_relevant_sentences(self, sentences: List[str]) -> List[str]:
        """过滤出与许可证条款相关的句子"""
        relevant_keywords = [
            'license', 'permit', 'grant', 'distribute', 'modify', 'use', 'commercial',
            'copyright', 'attribution', 'notice', 'source', 'binary', 'derivative',
            'sublicense', 'patent', 'trademark', 'warranty', 'liability', 'condition',
            'restriction', 'prohibition', 'exception', 'requirement', 'obligation'
        ]
        
        relevant_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # 过滤太短的句子
                continue
                
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in relevant_keywords):
                relevant_sentences.append(sentence)
        
        return relevant_sentences

    def analyze_sentence_ner_re(self, sentence: str) -> Dict[str, Any]:
        """分析单个句子的NER和RE"""
        # NER分析
        inputs = self.ner_tokenizer(sentence, return_tensors="pt", truncation=True, 
                                   max_length=512, padding="max_length")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            predictions_idx = self.ner_model.predict(input_ids, attention_mask)

        if predictions_idx is None:
            return {"sentence": sentence, "entities": [], "relations": []}

        pred_indices = predictions_idx[0].cpu().numpy()
        mask = attention_mask[0].cpu().numpy()
        tokens = self.ner_tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

        # 提取实体
        entities = self._extract_entities_from_tags(tokens, pred_indices, mask)
        
        # RE分析
        relations = self._extract_relations(sentence, entities)
        
        return {
            "sentence": sentence,
            "entities": entities,
            "relations": relations
        }

    def _extract_entities_from_tags(self, tokens, pred_indices, mask):
        """从NER标签中提取实体"""
        entities = []
        current_entity = None
        current_tokens = []
        start_index = -1

        for i, token in enumerate(tokens):
            if mask[i] == 0:
                break
            if token in [self.ner_tokenizer.cls_token, self.ner_tokenizer.sep_token, self.ner_tokenizer.pad_token]:
                continue
                
            pred_idx = pred_indices[i]
            tag = self.idx_to_tag.get(pred_idx, 'O')
            tag_class = tag.split('-')[0]
            tag_type = tag.split('-')[-1] if '-' in tag else None

            if tag_class == 'B':
                if current_entity:
                    entities.append({
                        "text": self.ner_tokenizer.convert_tokens_to_string(current_tokens),
                        "type": current_entity,
                        "start_token": start_index,
                        "end_token": i - 1
                    })
                current_entity = tag_type
                current_tokens = [token]
                start_index = i
            elif tag_class == 'I' and current_entity == tag_type:
                current_tokens.append(token)
            elif tag_class == 'O':
                if current_entity:
                    entities.append({
                        "text": self.ner_tokenizer.convert_tokens_to_string(current_tokens),
                        "type": current_entity,
                        "start_token": start_index,
                        "end_token": i - 1
                    })
                current_entity = None
                current_tokens = []
                start_index = -1

        if current_entity:
            entities.append({
                "text": self.ner_tokenizer.convert_tokens_to_string(current_tokens),
                "type": current_entity,
                "start_token": start_index,
                "end_token": len(tokens) - 1
            })

        return entities

    def _extract_relations(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        """提取实体间的关系"""
        relations = []
        
        # 生成实体对
        actions = [e for e in entities if e['type'] == 'Action']
        others = [e for e in entities if e['type'] in ['Recipient', 'Condition', 'Attitude']]

        for action in actions:
            for other in others:
                subj_text = action['text'].strip()
                obj_text = other['text'].strip()

                if not subj_text or not obj_text:
                    continue

                # 构建RE输入
                prompt = f"[sub] {subj_text} [sub] {self.re_tokenizer.mask_token} [obj] {obj_text} [obj] ."
                full_prompt_text = f"{sentence} {prompt}"

                inputs = self.re_tokenizer(
                    full_prompt_text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.re_args.max_seq_length,
                    return_tensors='pt'
                )

                # 确保mask token存在
                input_ids_list = inputs['input_ids'][0].tolist()
                if self.re_tokenizer.mask_token_id not in input_ids_list:
                    for i, token_id in enumerate(input_ids_list):
                        if token_id not in [self.re_tokenizer.cls_token_id, 
                                          self.re_tokenizer.sep_token_id, 
                                          self.re_tokenizer.pad_token_id]:
                            input_ids_list[i] = self.re_tokenizer.mask_token_id
                            inputs['input_ids'] = torch.tensor([input_ids_list])
                            break

                # 准备RE输入
                input_ids = inputs['input_ids'].squeeze(0)
                attention_mask = inputs['attention_mask'].squeeze(0)
                labels = torch.tensor([0])
                so = torch.tensor([[0, 0, 0, 0]])

                collated_batch = self.wiki80_processor._collate_fn_wiki80(
                    [(input_ids, attention_mask, labels, so)], self.re_tokenizer)

                # 运行RE预测
                with torch.no_grad():
                    re_preds_indices = predict_re(self.re_lit_model, [collated_batch])
                    predicted_relations = postprocess_predictions(re_preds_indices, self.id2rel)

                if predicted_relations and predicted_relations[0] not in ["Unknown", "no_relation"]:
                    relations.append({
                        "subject": action,
                        "object": other,
                        "relation": predicted_relations[0]
                    })

        return relations

    def analyze_license_comprehensive(self, license_text: str, output_file: str = None) -> List[Dict[str, Any]]:
        """综合分析许可证文本"""
        self.load_models()
        
        # 1. 智能分句
        sentences = self.split_license_text(license_text)
        print(f"文本分割为 {len(sentences)} 个相关句子")
        
        # 2. 批量分析每个句子
        all_results = []
        for i, sentence in enumerate(sentences):
            print(f"分析句子 {i+1}/{len(sentences)}: {sentence[:50]}...")
            result = self.analyze_sentence_ner_re(sentence)
            all_results.append(result)
        
        # 3. 保存结果
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {output_file}")
        
        return all_results

# 使用示例
def main():
    analyzer = ImprovedLicenseAnalyzer()
    
    # 读取许可证文件
    with open('D:/PythonProject/LiResolver_copy/license_txt/cherry-studio.txt', 'r', encoding='utf-8') as f:
        license_text = f.read()
    
    # 分析
    results = analyzer.analyze_license_comprehensive(
        license_text=license_text,
        output_file='D:/PythonProject/LiResolver_copy/license_analysis/improved_analysis_results.json'
    )
    
    # 打印结果摘要
    total_entities = sum(len(r['entities']) for r in results)
    total_relations = sum(len(r['relations']) for r in results)
    print(f"\n分析完成！")
    print(f"总句子数: {len(results)}")
    print(f"总实体数: {total_entities}")
    print(f"总关系数: {total_relations}")

if __name__ == "__main__":
    main()