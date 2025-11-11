import re
from collections import defaultdict

def count_entities(bio_text):
    entity_counts = defaultdict(int)
    current_entity = None
    
    lines = bio_text.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue
            
        parts = line.split()
        if len(parts) < 2:
            continue
            
        tag = parts[-1]
        
        # Check for B- and I- tags
        if tag.startswith('B-'):
            entity_type = tag[2:]
            entity_counts[entity_type] += 1
            current_entity = entity_type
        elif tag.startswith('I-'):
            # Only count if following a B- tag of the same type
            entity_type = tag[2:]
            if current_entity != entity_type:
                # Treat as new entity if not following matching B- tag
                entity_counts[entity_type] += 1
                current_entity = entity_type
        else:
            current_entity = None
    
    return entity_counts

def main():
    # Read the input file
    file_path = r"C:\Users\HYP\Desktop\experiment\ner_train_new.txt"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            bio_text = file.read()
        
        counts = count_entities(bio_text)
        
        print("\nEntity Counts:")
        for entity_type, count in counts.items():
            print(f"{entity_type}: {count}")
            
        total_entities = sum(counts.values())
        print(f"\nTotal entities: {total_entities}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()