import ijson
import json

def get_first_n_items(file_path, n=5):
    with open(file_path, encoding='utf-8') as f:  # 注意要使用二进制模式
        items = []
        # 假设JSON文件最外层是一个数组
        for i, item in enumerate(ijson.items(f, 'item')):
            if i >= n:
                break
            items.append(item)
        return items

if __name__ == "__main__":
    file_path = "./prompts/sql_synthesis_prompts.json"
    results = get_first_n_items(file_path, 50)
    output_file = "./prompts/test.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
