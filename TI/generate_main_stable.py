# 根据许可证analysis_results.json 生成summary_predict.csv
# 最终版 用这个！横向用的是这个
import json
import csv
import os
import re
import argparse # 用于接收命令行参数
from collections import defaultdict, OrderedDict # 添加 OrderedDict 导入

# --- 配置 ---
# 默认输入输出文件路径，可以通过命令行参数覆盖
DEFAULT_ANALYSIS_FILE = r'd:\PythonProject\LiResolver_copy\license_analysis\improved_analysis_results.json'
DEFAULT_OUTPUT_CSV = r'd:\PythonProject\LiResolver_copy\license_terms\summary_predict_cry_main.csv'
# 新增：证据输出文件路径
DEFAULT_EVIDENCE_FILE = r'd:\PythonProject\LiResolver_copy\license_terms\summary_evidence_cry_main.txt'

# 1. 定义目标条款 (保持不变)
TARGET_TERMS = [
    "Distribute", "Modify", "Commercial Use", "Hold Liable", "Include Copyright",
    "Include License", "Sublicense", "Use Trademark", "Private Use", "Disclose Source",
    "State Changes", "Place Warranty", "Include Notice", "Include Original", "Give Credit",
    "Use Patent Claims", "Rename", "Relicense", "Contact Author", "Include Install Instructions",
    "Compensate for Damages", "Statically Link", "Pay Above Use Threshold"
]

# 2. 定义映射规则 (保持不变，但需要持续优化)
TERM_KEYWORDS = {
    "Distribute": ['distribute', 'distribution', 'copy', 'copies', 'redistribute', 'propagate', 'convey', 'reproduction'], 
    "Modify": ['modify', 'modification', 'derivative works', 'adapt', 'change', 'changes', 'prepare derivative works'], 
    "Commercial Use": ['commercial', 'sell', 'sale', 'non-commercial', 'noncommercial', 'offer to sell', 'business', 
        'profit', 'fee', 'charge', 'monetary', 'revenue', 'income', 'commercial purpose', 
        'commercial advantage', 'commercial gain', 'commercial activity', 'commercial benefit', 
        'commercial exploitation', 'commercial distribution',
        # 新增Cherry Studio特定关键词
        'commercial authorization', 'commercial licensing', 'enterprise services', 
        'enterprise customers', 'hardware bundling', 'bundled sale', 'public cloud services',
        'large-scale procurement', 'government procurement', 'educational institutions',
        'commercial usage', 'cloud business operations', 'pre-install', 'integrate into hardware',
        'cumulative usage', 'users threshold', 'written authorization', 'explicit authorization'], 
    "Hold Liable": ['liable', 'liability', 'damages', 'warranty', 'disclaim'],
    # 扩展版权相关关键词
    "Include Copyright": ['copyright notice', 'copyright', 'copyright statement', 'notice of copyright', 'copyright holder', 'copyright owner'],
    # 扩展许可证相关关键词
    "Include License": ['license text', 'copy of the license', 'include license', 'license notice', 'license terms', 'license conditions', 'license file', 'license copy'],
    "Sublicense": ['sublicense'],
    "Use Trademark": ['trademark', 'service mark', 'logo', 'names','trademarks','marks','mark'],
    "Private Use": ['private use', 'internal use'],
    # 扩展源代码相关关键词
    "Disclose Source": ['source code', 'disclose source', 'provide source', 'complete source', 'corresponding source', 'source form', 'machine-readable form'],
    # 扩展变更相关关键词
    "State Changes": ['state changes', 'indicate changes', 'document changes', 'mark changes', 'identify changes', 'note changes', 'modification notice'],
    "Place Warranty": ['warranty', 'guarantee'],
    "Include Notice": ['notice', 'attribution notice', 'preserve notice', 'retain notice', 'maintain notice'],
    # 扩展原始代码相关关键词
    "Include Original": ['include original', 'original work', 'original code', 'original program', 'original copyright'],
    "Give Credit": ['credit', 'attribution'],
    "Use Patent Claims": ['patent', 'patent claims', 'patent rights', 'patent license'], 
    "Rename": ['rename'],
    "Relicense": ['relicense', 'change license'],
    "Contact Author": ['contact author'],
    "Include Install Instructions": ['installation instructions'],
    "Compensate for Damages": ['damages', 'compensate'],
    "Statically Link": ['static link', 'statically linked', 'link', 'bind'], 
    "Pay Above Use Threshold": ['pay', 'fee', 'threshold', 'charge', 'royalty-free'], 
}

# 确保关键词列表是有序的，以保证每次匹配顺序一致
PERMISSION_KEYWORDS = sorted(['may', 'permit', 'permitted', 'allow', 'allowed', 'grant', 'grants', 'authorized', 'can', 'hereby grants', 'no-charge', 'royalty-free'])
PROHIBITION_KEYWORDS = sorted(['must not', 'shall not', 'may not', 'cannot', 'prohibit', 'prohibited', 'forbidden', 'disallow', 'not authorized', 'void', 'terminate', 
                        'without permission', 'no ', 'not include', 'not permitted', 'restriction', 'restricted', 'limitation'])
# 将 'shall' 和 'must' 移到 OBLIGATION_KEYWORDS，因为它们主要表示义务
OBLIGATION_KEYWORDS = sorted(['must', 'shall', 'required', 'condition', 'conditioned upon', 
                       'obligation', 'obligated', 'need to', 'has to', 'have to', 'necessary', 'ensure', 'ensure that',
                       'make sure', 'preserve', 'maintain', 'keep', 'retain', 'include'])

# --- 函数定义 ---

def find_term_for_entity(entity_text):
    """根据实体文本查找对应的目标条款"""
    text_lower = entity_text.lower()
    matched_terms = []
    # 完全匹配优先
    for term in TARGET_TERMS:  # 使用有序的TARGET_TERMS列表
        keywords = TERM_KEYWORDS.get(term, [])
        if text_lower in keywords:
             matched_terms.append(term)
    # 如果没有完全匹配，尝试部分包含匹配 (更宽松，可能不准)
    if not matched_terms:
        for term in TARGET_TERMS:  # 使用有序的TARGET_TERMS列表
            keywords = TERM_KEYWORDS.get(term, [])
            for keyword in sorted(keywords):  # 确保关键词顺序一致
                # 确保 keyword 存在于 entity_text 中，并且 keyword 不是太短
                if keyword in text_lower and len(keyword) > 3:
                    matched_terms.append(term)
                    break  # 找到一个匹配就跳出内层循环，避免重复添加
    # 去重并保持顺序
    seen = set()
    matched_terms = [term for term in matched_terms if not (term in seen or seen.add(term))]
    # 如果匹配多个，可能需要更复杂的逻辑，这里简单返回第一个
    return matched_terms[0] if matched_terms else None


def determine_attitude(sentence_text, action_entity, entities, relations):
    """
    根据句子文本、动作实体、相关实体和关系判断态度。
    返回: 1 (允许/Can), 2 (禁止/Cannot), 3 (必须/Must), 0 (未知/Unknown)
    优先级: 禁止(2) > 必须(3) > 允许(1) > 未知(0)
    """
    attitude_result = 0 # Default to unknown
    sentence_lower = sentence_text.lower()
    action_text_lower = action_entity['text'].lower()
    action_start = action_entity['start_token']
    action_end = action_entity['end_token']

    # --- 增强关系处理 ---
    # 1. 直接从关系中提取态度
    related_attitudes = []
    related_conditions = []
    has_related_prohibition = False
    has_related_obligation = False
    has_related_permission = False
    
    # 记录关系类型和关联实体，用于更精确的态度判断
    relation_types = []
    attitude_entities = []
    condition_entities = []

    # 对关系进行排序，确保处理顺序一致
    sorted_relations = sorted(relations, key=lambda r: (r.get('type', ''), 
                                                       r.get('subject', {}).get('start_token', 0),
                                                       r.get('object', {}).get('start_token', 0)))

    for rel in sorted_relations:
        rel_type = rel.get('type', '')  # 获取关系类型
        rel_subject = rel.get('subject', {})
        rel_object = rel.get('object', {})
        
        # 检查关系的主语或宾语是否是当前处理的 Action 实体
        is_subject_match = (rel_subject.get('text') == action_entity['text'] and
                            rel_subject.get('start_token') == action_start and
                            rel_subject.get('end_token') == action_end)

        is_object_match = (rel_object.get('text') == action_entity['text'] and
                           rel_object.get('start_token') == action_start and
                           rel_object.get('end_token') == action_end)

        if is_subject_match or is_object_match:
            # 记录关系类型
            relation_types.append(rel_type)
            
            # 确定哪个是 Action，哪个是关联实体 (Attitude/Condition)
            linked_entity = rel_object if is_subject_match else rel_subject
            linked_type = linked_entity.get('type')
            linked_text = linked_entity.get('text', '').lower()

            if linked_type == 'Attitude':
                related_attitudes.append(linked_text)
                attitude_entities.append(linked_entity)
                
                # 直接从关系类型判断态度 - 使用固定顺序的列表检查
                prohibition_rel_types = ['prohibit', 'prohibits', 'prohibited', 'forbid', 'forbids', 'forbidden']
                obligation_rel_types = ['require', 'requires', 'required', 'must', 'shall', 'obligate', 'obligates']
                permission_rel_types = ['permit', 'permits', 'permitted', 'allow', 'allows', 'allowed', 'grant', 'grants']
                
                if rel_type in prohibition_rel_types:
                    has_related_prohibition = True
                elif rel_type in obligation_rel_types:
                    has_related_obligation = True
                elif rel_type in permission_rel_types:
                    has_related_permission = True
                
                # 如果关系类型不明确，再从态度实体文本中判断
                if not (has_related_prohibition or has_related_obligation or has_related_permission):
                    # 使用有序的关键词列表
                    if any(keyword in linked_text for keyword in PROHIBITION_KEYWORDS):
                        has_related_prohibition = True
                    elif any(keyword in linked_text for keyword in OBLIGATION_KEYWORDS):
                        has_related_obligation = True
                    elif any(keyword in linked_text for keyword in PERMISSION_KEYWORDS):
                        has_related_permission = True

            elif linked_type == 'Condition':
                related_conditions.append(linked_text)
                condition_entities.append(linked_entity)
                # 分析条件中是否包含态度信息
                if any(keyword in linked_text for keyword in OBLIGATION_KEYWORDS):
                    # 条件中包含义务关键词，可能是必须满足的条件
                    has_related_obligation = True

    # 根据关联实体的判断结果确定态度 (应用优先级)
    if has_related_prohibition:
        return 2 # 禁止优先
    elif has_related_obligation:
        return 3 # 其次是必须
    elif has_related_permission:
        attitude_result = 1 # 允许
    
    # 2. 分析关系类型本身
    # 即使没有明确的态度实体，关系类型本身也可能暗示态度
    if attitude_result == 0:
        # 使用固定顺序的列表检查
        prohibition_rel_types = ['prohibit', 'prohibits', 'prohibited', 'forbid', 'forbids', 'forbidden']
        obligation_rel_types = ['require', 'requires', 'required', 'must', 'shall', 'obligate', 'obligates']
        permission_rel_types = ['permit', 'permits', 'permitted', 'allow', 'allows', 'allowed', 'grant', 'grants']
        
        if any(rel_type in prohibition_rel_types for rel_type in relation_types):
            return 2  # 禁止
        elif any(rel_type in obligation_rel_types for rel_type in relation_types):
            return 3  # 必须
        elif any(rel_type in permission_rel_types for rel_type in relation_types):
            attitude_result = 1  # 允许
    
    # 3. 分析条件实体
    # 如果有条件但没有明确态度，进一步分析条件
    if attitude_result == 0 and condition_entities:
        # 检查条件是否暗示某种态度
        condition_texts = [entity.get('text', '').lower() for entity in condition_entities]
        combined_condition = ' '.join(condition_texts)
        
        if any(keyword in combined_condition for keyword in PROHIBITION_KEYWORDS):
            return 2  # 禁止
        elif any(keyword in combined_condition for keyword in OBLIGATION_KEYWORDS):
            return 3  # 必须
        elif any(keyword in combined_condition for keyword in PERMISSION_KEYWORDS):
            attitude_result = 1  # 允许
        elif related_conditions:  # 有条件但没有明确态度，可能是允许的
            attitude_result = 1  # 默认为允许，但可以根据具体许可证调整

    # --- 如果 Relations 未明确禁止或必须，检查句子级关键词 ---
    # 仅在没有通过 Relations 确定为 2 或 3 时执行
    if attitude_result != 2 and attitude_result != 3:
        # 简化：检查整个句子中是否有关键词，并尝试关联
        sentence_has_prohibition = any(keyword in sentence_lower for keyword in PROHIBITION_KEYWORDS)
        sentence_has_obligation = any(keyword in sentence_lower for keyword in OBLIGATION_KEYWORDS)
        sentence_has_permission = any(keyword in sentence_lower for keyword in PERMISSION_KEYWORDS)

        # 查找与 Action 距离最近的关键词
        min_dist = float('inf')
        closest_kw_type = None # 'prohibit', 'obligate', 'permit'

        # 使用固定顺序的关键词类型列表
        keyword_types = [('prohibit', PROHIBITION_KEYWORDS),
                         ('obligate', OBLIGATION_KEYWORDS),
                         ('permit', PERMISSION_KEYWORDS)]

        for kw_type, keywords in keyword_types:
            for keyword in keywords:  # 已排序的关键词列表
                try:
                    # 使用 re.finditer 查找所有匹配项
                    for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', sentence_lower):
                        kw_index = match.start()
                        act_index = sentence_lower.find(action_text_lower) # 查找 Action 第一次出现的位置
                        if act_index != -1:
                            dist = abs(kw_index - act_index)
                            if dist < min_dist:
                                min_dist = dist
                                closest_kw_type = kw_type
                except re.error: # 处理可能的正则表达式错误
                     # 如果关键词不是有效的正则表达式部分，使用简单查找
                     if keyword in sentence_lower:
                         kw_index = sentence_lower.find(keyword)
                         act_index = sentence_lower.find(action_text_lower)
                         if act_index != -1:
                             dist = abs(kw_index - act_index)
                             if dist < min_dist:
                                 min_dist = dist
                                 closest_kw_type = kw_type

        # 根据最近的关键词类型判断 (设置一个距离阈值，例如 50 字符)
        if min_dist < 50:
            if closest_kw_type == 'prohibit':
                 return 2 # 禁止优先
            elif closest_kw_type == 'obligate':
                 return 3 # 其次是必须
            elif closest_kw_type == 'permit':
                 # 只有在之前结果是 0 时才更新为 1
                 if attitude_result == 0:
                     attitude_result = 1

    # --- 处理特殊条款逻辑 ---
    term = find_term_for_entity(action_entity['text'])
    
    # 特殊条款处理 - 使用固定顺序的关键词列表
    if term == "Hold Liable":
        disclaimer_keywords = sorted(['disclaim', 'disclaimer', 'as is', 'without warranty', 'not liable'])
        liability_keywords = sorted(['liable for', 'responsible for'])
        
        if any(k in sentence_lower for k in disclaimer_keywords):
            return 2 # 不允许追究责任
        if any(k in sentence_lower for k in liability_keywords):
            return 1 # 允许追究责任
    elif term == "Place Warranty":
        disclaimer_keywords = sorted(['disclaim', 'disclaimer', 'as is', 'without warranty', 'no warranty'])
        warranty_keywords = sorted(['provide warranty', 'guarantee'])
        
        if any(k in sentence_lower for k in disclaimer_keywords):
            return 2 # 不允许提供质保
        if any(k in sentence_lower for k in warranty_keywords):
            return 1 # 允许提供质保
    elif term == "Pay Above Use Threshold":
        if 'royalty-free' in sentence_lower or 'no-charge' in sentence_lower:
             return 1 # 明确免费，视为允许（在0费用下使用）
    
    # 通用的"必须"类条款处理逻辑
    # 这些条款在许多许可证中通常是必须的
    must_terms = ["Include Copyright", "Include License", "Include Notice", "Include Original", 
                  "Disclose Source", "State Changes", "Give Credit"]
    
    if term in must_terms:
        # 检查句子中是否有义务词靠近这些动作
        for keyword in OBLIGATION_KEYWORDS:
            if keyword in sentence_lower:
                kw_index = sentence_lower.find(keyword)
                act_index = sentence_lower.find(action_text_lower)
                # 使用更宽松的距离阈值，适应不同许可证的表达方式
                if act_index != -1 and abs(kw_index - act_index) < 50:
                    # 检查是否有否定词修饰义务词
                    neg_words = sorted(["not", "no", "never", "without"])
                    has_negation = any(sentence_lower.rfind(neg, max(0, kw_index-10), kw_index) != -1 for neg in neg_words)
                    if not has_negation:
                        return 3  # 必须
    
    # 通用的句子模式检测 - 使用固定顺序的模式列表
    # 检查常见的必须性表述模式
    obligation_patterns = [
        r"you must\s+\w*\s*" + re.escape(action_text_lower),
        r"shall\s+\w*\s*" + re.escape(action_text_lower),
        r"required to\s+\w*\s*" + re.escape(action_text_lower),
        r"need to\s+\w*\s*" + re.escape(action_text_lower),
        r"have to\s+\w*\s*" + re.escape(action_text_lower),
        # 反向模式 - 动作在前，义务词在后
        re.escape(action_text_lower) + r"\s+\w*\s*must",
        re.escape(action_text_lower) + r"\s+\w*\s*shall",
        re.escape(action_text_lower) + r"\s+\w*\s*required",
        # 条件模式
        r"provided that\s+\w*\s*" + re.escape(action_text_lower),
        r"on condition that\s+\w*\s*" + re.escape(action_text_lower),
        r"subject to\s+\w*\s*" + re.escape(action_text_lower)
    ]
    
    for pattern in obligation_patterns:
        if re.search(pattern, sentence_lower):
            return 3  # 必须
    
    # 检查常见的禁止性表述模式
    prohibition_patterns = [
        r"you may not\s+\w*\s*" + re.escape(action_text_lower),
        r"shall not\s+\w*\s*" + re.escape(action_text_lower),
        r"must not\s+\w*\s*" + re.escape(action_text_lower),
        r"cannot\s+\w*\s*" + re.escape(action_text_lower),
        r"prohibited from\s+\w*\s*" + re.escape(action_text_lower)
    ]
    
    for pattern in prohibition_patterns:
        if re.search(pattern, sentence_lower):
            return 2  # 禁止
    
    # 检查常见的允许性表述模式
    permission_patterns = [
        r"you may\s+\w*\s*" + re.escape(action_text_lower),
        r"are permitted to\s+\w*\s*" + re.escape(action_text_lower),
        r"are allowed to\s+\w*\s*" + re.escape(action_text_lower),
        r"grant\w* you\s+\w*\s*" + re.escape(action_text_lower),
        r"right to\s+\w*\s*" + re.escape(action_text_lower)
    ]
    
    for pattern in permission_patterns:
        if re.search(pattern, sentence_lower):
            return 1  # 允许

    # 返回最终判断结果
    return attitude_result

def map_sentence_to_terms(sentence_data):
    """
    处理单个句子的数据，映射到目标条款和态度。
    返回一个字典: {term: {'attitude': 0|1|2|3, 'evidence': sentence_text | None}}
    """
    # 使用OrderedDict确保结果顺序一致
    sentence_results = OrderedDict((term, {'attitude': 0, 'evidence': None}) for term in TARGET_TERMS) # 初始化
    sentence_text = sentence_data.get('sentence', '')
    entities = sentence_data.get('entities', [])
    relations = sentence_data.get('relations', [])

    if not entities or not sentence_text:
        return sentence_results

    # 对实体进行排序，确保处理顺序一致
    action_entities = sorted([e for e in entities if e.get('type') == 'Action'], 
                            key=lambda e: (e.get('start_token', 0), e.get('end_token', 0)))

    for action_entity in action_entities:
        target_term = find_term_for_entity(action_entity['text'])
        if target_term:
            attitude = determine_attitude(sentence_text, action_entity, entities, relations)
            if attitude != 0:
                # 如果当前句子判断出态度，记录下来
                # 注意：一个句子可能对同一个条款有多次提及，这里简单覆盖
                sentence_results[target_term]['attitude'] = attitude
                sentence_results[target_term]['evidence'] = sentence_text.strip() # 存储句子原文

    return sentence_results

# ... 前面的代码保持不变 ...

def aggregate_results_for_license(all_sentence_data):
    """
    聚合单个许可证所有句子的结果。
    新的聚合逻辑：优先考虑"允许"，然后是"必须"，最后是"禁止"。
    返回一个字典: {term: {'attitude': 0|1|2|3, 'evidence': [sentence1, sentence2, ...]}}
    """
    # 使用OrderedDict确保结果顺序一致
    from collections import OrderedDict
    
    # 初始化，每个条款先假设一个"最不优先"的态度（未知）
    # 并且为每种可能的最终态度预留证据列表
    final_results = OrderedDict()
    for term in TARGET_TERMS:
        final_results[term] = {
            'attitude': 0, 
            'evidence_can': set(), 
            'evidence_must': set(), 
            'evidence_cannot': set(), 
            'evidence_unknown': set()
        }

    if not all_sentence_data:
        print("  [Warning] No sentence data provided for aggregation.")
        # 返回转换后的格式
        return OrderedDict((term, {'attitude': data['attitude'], 'evidence': sorted(list(data['evidence_unknown']))}) 
                          for term, data in final_results.items())

    # 收集所有句子对每个条款的判断
    term_sentence_judgments = OrderedDict((term, []) for term in TARGET_TERMS)
    
    # 对句子数据进行排序，确保处理顺序一致
    sorted_sentence_data = sorted(all_sentence_data, 
                                 key=lambda s: s.get('sentence_id', 0) if isinstance(s.get('sentence_id'), int) else 0)
    
    for sentence_data in sorted_sentence_data:
        sentence_mapping = map_sentence_to_terms(sentence_data)
        for term, data in sentence_mapping.items():
            if data['attitude'] != 0 and data['evidence']:
                term_sentence_judgments[term].append({'attitude': data['attitude'], 'evidence': data['evidence']})

    # 应用新的聚合逻辑
    for term in TARGET_TERMS:
        judgments = term_sentence_judgments[term]
        if not judgments:
            final_results[term]['evidence_unknown'].add("(No specific sentence found for this term)")
            continue

        has_can = any(j['attitude'] == 1 for j in judgments)
        has_must = any(j['attitude'] == 3 for j in judgments)
        has_cannot = any(j['attitude'] == 2 for j in judgments)

        current_evidence_set = set()

        if has_can:
            final_results[term]['attitude'] = 1
            for j in judgments: # 收集所有相关的证据，包括允许、必须（作为条件）
                if j['attitude'] == 1 or j['attitude'] == 3: # 允许的证据和作为条件的必须证据
                    current_evidence_set.add(j['evidence'])
                # 也可以考虑是否加入相关的禁止性条件作为补充说明
                # elif j['attitude'] == 2:
                # current_evidence_set.add(f"[Conditional Prohibition context]: {j['evidence']}")
        elif has_must: # 没有明确的 can，但有 must
            final_results[term]['attitude'] = 3
            for j in judgments:
                if j['attitude'] == 3:
                    current_evidence_set.add(j['evidence'])
        elif has_cannot: # 没有 can 和 must，但有 cannot
            final_results[term]['attitude'] = 2
            for j in judgments:
                if j['attitude'] == 2:
                    current_evidence_set.add(j['evidence'])
        else: # 所有判断都是 0 (或者没有有效判断)
            final_results[term]['attitude'] = 0
            for j in judgments: # 应该不会执行到这里，因为前面有 if not judgments
                current_evidence_set.add(j['evidence'])
            if not current_evidence_set:
                 current_evidence_set.add("(No specific sentence with attitude found, though term mentioned)")

        # 根据最终确定的态度，选择对应的证据集合
        if final_results[term]['attitude'] == 1:
            final_results[term]['evidence_can'] = current_evidence_set
        elif final_results[term]['attitude'] == 3:
            final_results[term]['evidence_must'] = current_evidence_set
        elif final_results[term]['attitude'] == 2:
            final_results[term]['evidence_cannot'] = current_evidence_set
        else: # attitude == 0
            # 如果上面逻辑正确，这里的 evidence_unknown 应该已经被填充
            if not current_evidence_set and not final_results[term]['evidence_unknown']:
                 final_results[term]['evidence_unknown'].add("(No specific sentence found for this term's attitude)")
            elif current_evidence_set: # 如果因为某种原因 current_evidence_set 有内容但态度为0
                 final_results[term]['evidence_unknown'].update(current_evidence_set)

    # 转换输出格式，只保留与最终态度相关的证据
    output_results = OrderedDict()
    for term in TARGET_TERMS:
        data = final_results[term]
        final_attitude = data['attitude']
        evidence_to_use = set()
        if final_attitude == 1:
            evidence_to_use = data['evidence_can']
            # 对于允许的情况，也把相关的"必须"作为条件证据加入
            for j in term_sentence_judgments[term]:
                if j['attitude'] == 3:
                    evidence_to_use.add(f"[Condition/Obligation]: {j['evidence']}")
        elif final_attitude == 3:
            evidence_to_use = data['evidence_must']
        elif final_attitude == 2:
            evidence_to_use = data['evidence_cannot']
        else: # attitude == 0
            evidence_to_use = data['evidence_unknown']
        
        # 如果没有任何证据，但态度非0（例如通过默认逻辑），给一个通用提示
        if not evidence_to_use and final_attitude != 0:
            evidence_to_use.add("(Attitude determined by general rule or implicit logic, no specific sentence)")
        elif not evidence_to_use and final_attitude == 0:
            evidence_to_use.add("(No specific sentence found for this term)")

        output_results[term] = {
            'attitude': final_attitude,
            'evidence': sorted(list(evidence_to_use))
        }
    return output_results
def extract_commercial_conditions(all_sentence_data, commercial_attitude):
    """
    提取与商用相关的条件限制
    
    参数:
        all_sentence_data: 所有句子的分析数据
        commercial_attitude: 商用的总体态度(0,1,2,3)
        
    返回:
        list: 商用条件限制列表
    """
    commercial_conditions = []
    
    # 如果不允许商用，直接返回空列表
    if commercial_attitude == 2:
        return ["不允许商业使用"]
    
    # 如果未明确是否可以商用，返回相应信息
    if commercial_attitude == 0:
        return ["许可证未明确说明是否允许商业使用"]
    
    # 查找与商用相关的条件
    for sentence_data in all_sentence_data:
        sentence_text = sentence_data.get('sentence', '')
        entities = sentence_data.get('entities', [])
        relations = sentence_data.get('relations', [])
        
        # 查找商用相关的Action实体
        commercial_actions = []
        for entity in entities:
            if entity.get('type') == 'Action':
                action_text = entity.get('text', '').lower()
                if any(keyword in action_text for keyword in TERM_KEYWORDS['Commercial Use']):
                    commercial_actions.append(entity)
        
        # 如果找到商用相关的Action，查找与之相关的条件
        for action in commercial_actions:
            # 查找与该Action相关的Condition实体
            related_conditions = []
            for relation in relations:
                subject = relation.get('subject', {})
                object_entity = relation.get('object', {})
                relation_type = relation.get('relation', '')
                
                # 检查关系的主语是否是当前的商用Action
                if (subject.get('text') == action.get('text') and 
                    subject.get('start_token') == action.get('start_token')):
                    # 如果宾语是Condition类型，添加到相关条件
                    if object_entity.get('type') == 'Condition':
                        related_conditions.append({
                            'condition_text': object_entity.get('text'),
                            'relation_type': relation_type,
                            'sentence': sentence_text
                        })
                
                # 检查关系的宾语是否是当前的商用Action
                elif (object_entity.get('text') == action.get('text') and 
                      object_entity.get('start_token') == action.get('start_token')):
                    # 如果主语是Condition类型，添加到相关条件
                    if subject.get('type') == 'Condition':
                        related_conditions.append({
                            'condition_text': subject.get('text'),
                            'relation_type': relation_type,
                            'sentence': sentence_text
                        })
            
            # 如果找到相关条件，添加到商用条件列表
            for condition in related_conditions:
                commercial_conditions.append(condition)
            
            # 如果没有通过关系找到条件，但句子中包含条件关键词，也添加到商用条件列表
            if not related_conditions and any(keyword in sentence_text.lower() for keyword in ['provided', 'condition', 'if', 'when', 'unless', 'except', 'subject to']):
                commercial_conditions.append({
                    'condition_text': '(Implicit condition)',
                    'relation_type': 'implicit',
                    'sentence': sentence_text
                })
    
    # 去重并格式化条件
    unique_conditions = []
    seen_sentences = set()
    
    for condition in commercial_conditions:
        sentence = condition.get('sentence')
        if sentence not in seen_sentences:
            seen_sentences.add(sentence)
            unique_conditions.append(sentence)
    
    return unique_conditions if unique_conditions else ["允许商业使用，无明确条件限制"]
# 供调用
def analyze_license_terms(analysis_data, license_name=None, output_csv=None, output_evidence=None):
    """
    分析许可证条款并生成摘要结果
    
    参数:
        analysis_data: 许可证分析结果数据，可以是JSON文件路径或已加载的数据列表
        license_name: 许可证名称，如果为None则尝试从文件名推断
        output_csv: 输出CSV文件路径，如果为None则不保存CSV
        output_evidence: 输出证据文件路径，如果为None则不保存证据
        
    返回:
        dict: 包含许可证条款分析结果的字典，格式为 {term: {'attitude': 0|1|2|3, 'evidence': [...]}}
    """
    all_sentence_data = []
    
    # 处理输入数据
    if isinstance(analysis_data, str):
        # 如果是字符串，假定为文件路径
        try:
            with open(analysis_data, 'r', encoding='utf-8') as f:
                analysis_json = json.load(f)
                print(f"[Info] Successfully loaded JSON data from: {analysis_data}")
                
                # 检查顶层是否为列表
                if isinstance(analysis_json, list):
                    all_sentence_data = analysis_json
                    print(f"[Info] Found {len(all_sentence_data)} sentence analysis entries.")
                else:
                    print(f"[Error] Expected JSON top-level structure to be a list, but found {type(analysis_json)}. Cannot process.")
                    return None
                    
            # 如果未提供license_name，从文件名推断
            if license_name is None:
                base_name = os.path.basename(analysis_data)
                license_name, _ = os.path.splitext(base_name)
                # 简单清理常见的后缀
                license_name = license_name.replace('_analysis', '').replace('_result', '').replace('analysis_', '').replace('result_', '')
                print(f"[Info] License name not provided, derived from input file: '{license_name}'")
                
        except FileNotFoundError:
            print(f"[Error] Input analysis file not found: {analysis_data}")
            return None
        except json.JSONDecodeError as e:
            print(f"[Error] Failed to parse JSON file: {analysis_data}. Error: {e}")
            return None
        except Exception as e:
            print(f"[Error] An unexpected error occurred while reading the file: {e}")
            return None
    elif isinstance(analysis_data, list):
        # 如果已经是列表，直接使用
        all_sentence_data = analysis_data
        
        # 确保license_name存在
        if license_name is None:
            license_name = "Unknown_License"
            print(f"[Warning] No license name provided, using default: '{license_name}'")
    else:
        print(f"[Error] Invalid analysis_data type: {type(analysis_data)}. Expected string (file path) or list.")
        return None
    
    # 如果没有数据，返回None
    if not all_sentence_data:
        print("[Warning] No sentence data found. Cannot analyze license terms.")
        return None
    
    # 对整个文件（单个许可证）进行聚合
    print(f"[Info] Aggregating results for license: {license_name}")
    aggregated_summary = aggregate_results_for_license(all_sentence_data)
    if "Distribute" in aggregated_summary and aggregated_summary['Distribute']['attitude'] == 3:
            aggregated_summary['Distribute']['attitude'] = 1
    if "Modify" in aggregated_summary and aggregated_summary['Modify']['attitude'] == 3:
            aggregated_summary['Modify']['attitude'] = 1
    if "Commercial Use" in aggregated_summary and aggregated_summary['Commercial Use']['attitude'] == 3:
            aggregated_summary['Commercial Use']['attitude'] = 1
    if "Sublicense" in aggregated_summary and aggregated_summary['Sublicense']['attitude'] == 3:
            aggregated_summary['Sublicense']['attitude'] = 1
    aggregated_summary['Pay Above Use Threshold']['attitude'] = 0
    commercial_attitude = aggregated_summary.get('Commercial Use', {}).get('attitude', 0)
    commercial_conditions = extract_commercial_conditions(all_sentence_data, commercial_attitude)
    
    # 将商用条件添加到结果中
    aggregated_summary['Commercial Use']['conditions'] = commercial_conditions
    # 输出调试信息
    if "Modify" in aggregated_summary:
        print("-" * 20)
        print(f"[Debug] Final aggregated result for 'Modify':")
        print(f"  Attitude: {aggregated_summary['Modify']['attitude']}")
        print(f"  Evidence Count: {len(aggregated_summary['Modify']['evidence'])}")
        for ev in aggregated_summary['Modify']['evidence']:
            print(f"  Evidence Sentence: {ev}")
        print("-" * 20)
    else:
        print("[Debug] 'Modify' term not found in aggregated summary.")
    
    # 如果需要，保存CSV文件
    if output_csv:
        # 准备写入CSV的数据
        summary_row_attitude = {'': license_name}
        for term, data in aggregated_summary.items():
            summary_row_attitude[term] = data['attitude']
        all_license_summaries_attitude = [summary_row_attitude]
        
        # 写入CSV文件
        fieldnames = [''] + TARGET_TERMS  # CSV Header
        try:
            output_dir = os.path.dirname(output_csv)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"[Info] Created output directory: {output_dir}")
                
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_license_summaries_attitude)
            print(f"[Success] Successfully wrote attitude summary for '{license_name}' to: {output_csv}")
        except IOError as e:
            print(f"[Error] Failed to write CSV file: {output_csv}. Error: {e}")
        except Exception as e:
            print(f"[Error] An unexpected error occurred while writing the CSV: {e}")
    
    # 如果需要，保存证据文件
    if output_evidence:
        try:
            output_dir = os.path.dirname(output_evidence)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"[Info] Created output directory for evidence file: {output_dir}")
                
            with open(output_evidence, 'w', encoding='utf-8') as ev_file:
                ev_file.write(f"Evidence for License: {license_name}\n")
                ev_file.write("=" * 30 + "\n\n")
                attitude_map = {0: "Unknown", 1: "Can", 2: "Cannot", 3: "Must"}
                
                # 使用TARGET_TERMS确保顺序一致
                for term in TARGET_TERMS:
                    data = aggregated_summary[term]
                    attitude = data['attitude']
                    evidence_list = data['evidence']
                    if attitude != 0:  # 只输出有明确态度的条款
                        ev_file.write(f"Term: {term}\n")
                        ev_file.write(f"Attitude: {attitude} ({attitude_map[attitude]})\n")
                        ev_file.write("Evidence:\n")
                        if evidence_list:
                            for idx, sentence in enumerate(evidence_list):
                                ev_file.write(f"  [{idx+1}] {sentence}\n")
                        else:
                            ev_file.write("  (No specific sentence found as evidence)\n")
                        ev_file.write("-" * 20 + "\n")
                        ev_file.write("\n" + "=" * 30 + "\n")
                ev_file.write("商业使用条件限制:\n")
                if commercial_conditions:
                    for idx, condition in enumerate(commercial_conditions):
                        ev_file.write(f"  [{idx+1}] {condition}\n")
                else:
                    ev_file.write("  (未找到明确的商用条件限制)\n")
                ev_file.write("=" * 30 + "\n")
            print(f"[Success] Successfully wrote evidence summary for '{license_name}' to: {output_evidence}")
        except IOError as e:
            print(f"[Error] Failed to write evidence file: {output_evidence}. Error: {e}")
        except Exception as e:
            print(f"[Error] An unexpected error occurred while writing the evidence file: {e}")
    
    # 返回聚合结果
    return aggregated_summary


# --- 主程序 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize license terms from analysis results.')
    parser.add_argument('-i', '--input', default=DEFAULT_ANALYSIS_FILE,
                        help=f'Path to the input JSON analysis file (default: {DEFAULT_ANALYSIS_FILE})')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_CSV,
                        help=f'Path to the output CSV summary file (default: {DEFAULT_OUTPUT_CSV})')
    # 新增：证据文件参数
    parser.add_argument('-e', '--evidence', default=DEFAULT_EVIDENCE_FILE,
                        help=f'Path to the output evidence text file (default: {DEFAULT_EVIDENCE_FILE})')
    parser.add_argument('-n', '--name', default=None,
                        help='License name (if not provided, derived from input filename)')

    args = parser.parse_args()

    input_file = args.input
    output_csv_file = args.output
    output_evidence_file = args.evidence # 获取证据文件路径
    license_name = args.name

    # 推断 license_name 的逻辑
    if not license_name:
        base_name = os.path.basename(input_file)
        license_name, _ = os.path.splitext(base_name)
        # 简单清理常见的后缀
        license_name = license_name.replace('_analysis', '').replace('_result', '').replace('analysis_', '').replace('result_', '')
        print(f"[Info] License name not provided, derived from input file: '{license_name}'")

    # 调用分析函数处理许可证
    analyze_license_terms(input_file, license_name, output_csv_file, output_evidence_file)