import re

def sanitize_identifier(identifier: str) -> str:
    """
    清理并安全地引用 SQL 标识符。

    处理逻辑：
    1. 将输入转换为字符串，以防传入非字符串类型。
    2. 替换空格、括号及其他所有非字母、非数字、非下划线的字符为下划线。
    3. 检查清理后的标识符是否以字母开头，如果不是，则添加 'fld_' 前缀。
    4. 检查是否为 SQLite-VEC 的保留关键字 'distance'，如果是则重命名。
    5. 用双引号包裹最终结果，使其成为一个安全的 SQL 标识符。
    """
    # 确保输入是字符串
    s = str(identifier)
    
    # 替换空格和括号
    s = s.replace(' ', '_').replace('(', '').replace(')', '')
    
    # 替换所有非字母、非数字、非下划线的字符为下划线
    # 这个表达式会正确处理 Unicode 字符（如 ñ），将它们替换为 _
    # 如果你想保留 ñ 这样的字符，可以使用 re.sub(r'[^\w]', '_', s)
    s = re.sub(r'[^a-zA-Z0-9_]', '_', s)
    
    # 如果标识符不以字母开头，则添加前缀。
    # 这样可以同时处理以数字、下划线或其他符号开头的情况。
    if not re.match(r'^[a-zA-Z]', s):
        s = 'fld_' + s
        
    # 检查是否为 sqlite-vec 的保留关键字 'distance' (不区分大小写)
    if s.lower() == 'distance':
        s = 'distance_val'  # 重命名冲突列
        
    # 用双引号包裹，这是 SQL 标准的做法
    return f'"{s}"'

# --- 测试案例 ---
test_cases = [
    "ValidIdentifier",
    "Name (Full)",
    "2025_Sales",
    "_internal_id",
    "Treyes Albarracán", # 假设输入是正确的 UTF-8 字符串
    "some/field-name%",
    "distance",
    "Distance"
]

print("清理前的标识符 -> 清理后的 SQL 标识符")
print("-" * 40)
for case in test_cases:
    print(f"{case:<20} -> {sanitize_identifier(case)}")
