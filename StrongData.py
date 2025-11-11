import random
import copy
# from .data_utils import CoNLLDataset

def augment_dataset(dataset, augmentation_factor=1.5):
    """
    对数据集进行增强
    
    Args:
        dataset: CoNLLDataset实例
        augmentation_factor: 增强倍数
    
    Returns:
        augmented_data: 增强后的数据
    """
    original_data = dataset.data
    augmented_data = copy.deepcopy(original_data)
    
    # 计算需要增强的样本数
    num_to_augment = int(len(original_data) * (augmentation_factor - 1))
    
    for _ in range(num_to_augment):
        # 随机选择一个样本
        sample_idx = random.randint(0, len(original_data) - 1)
        words, tags = original_data[sample_idx]
        
        # 应用随机增强技术
        aug_words, aug_tags = apply_augmentation(words, tags)
        augmented_data.append((aug_words, aug_tags))
    
    return augmented_data

def apply_augmentation(words, tags):
    """应用随机增强技术"""
    aug_words = words.copy()
    aug_tags = tags.copy()
    
    # 随机选择一种增强方法
    aug_method = random.choice([
        'synonym_replacement',
        'random_insertion',
        'random_swap',
        'random_deletion'
    ])
    
    if aug_method == 'synonym_replacement':
        # 替换非实体词为同义词（简化版）
        for i in range(len(aug_words)):
            if aug_tags[i] == 'O' and random.random() < 0.1:
                aug_words[i] = aug_words[i] + "_syn"  # 简化版，实际应使用同义词库
    
    elif aug_method == 'random_insertion':
        # 在非实体位置随机插入词
        if len(aug_words) > 3:
            insert_pos = random.randint(0, len(aug_words) - 1)
            if aug_tags[insert_pos] == 'O':
                aug_words.insert(insert_pos, "inserted_word")
                aug_tags.insert(insert_pos, "O")
    
    elif aug_method == 'random_swap':
        # 交换两个非实体词
        if len(aug_words) > 3:
            non_entity_indices = [i for i, tag in enumerate(aug_tags) if tag == 'O']
            if len(non_entity_indices) >= 2:
                idx1, idx2 = random.sample(non_entity_indices, 2)
                aug_words[idx1], aug_words[idx2] = aug_words[idx2], aug_words[idx1]
    
    elif aug_method == 'random_deletion':
        # 随机删除非实体词
        i = 0
        while i < len(aug_words):
            if aug_tags[i] == 'O' and random.random() < 0.1:
                del aug_words[i]
                del aug_tags[i]
            else:
                i += 1
    
    return aug_words, aug_tags