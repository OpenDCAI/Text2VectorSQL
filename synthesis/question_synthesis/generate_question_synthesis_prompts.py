import json
import os
import random
import sqlite3
import sqlite_vec
import sqlite_lembed
import numpy as np
import re
from tqdm import tqdm

# 风格描述 (无需改动)
style2desc = {
"Formal": '''**Formal Style**
   - Uses standard grammar and vocabulary.
   - Example: Find all students older than 18 years and return their home addresses.
   - Vector Example: Find the three articles most closely related to Stable Diffusion and return them.''',

"Colloquial": '''**Colloquial Style**
   - Employs informal vocabulary and expressions.
   - Example: Hey! Could you help me find all the students who are over 18? I'd love to know their names and where they live.
   - Vector Example: Hey there! Can you grab me the top 3 articles that are most closely related to Stable Diffusion?''',

"Imperative": '''**Imperative Style**
   - Uses command or directive sentences.
   - Example: Could you please gather all the students who are older than 18? I really need to know their names and where they live!
   - Vector Example: Please find the three articles most closely related to Stable Diffusion and return their name.''',

"Interrogative": '''**Interrogative Style**
   - Uses question forms.
   - Example: Could you tell me which students are older than 18 and what their home addresses are?
   - Vector Example: Could you show me the 3 articles that most have to do with Stable Diffusion?''',

"Descriptive": '''**Descriptive Style**
   - Uses detailed descriptions with contextual information.
   - Example: I want to know the names and home addresses of all students older than 18.
   - Vector Example: I need to find articles that most closely related to Stable Diffusion, returning the top 3 matches sorted by cosine similarity.''',

"Concise": '''**Concise Style**
   - Use short sentences.
   - Example: Students older than 18, return their names and addresses.
   - Vector Example: Top 3 related articles to Stable Diffusion.''',

"Vague": '''**Vague Style**
   - Includes ambiguous vocabulary requiring inference.
   - Example: What are the names and addresses of those older students? (External Knowledge: 'older students' refers to age >= 18.)
   - Vector Example: Find a few articles have to do with Stable Diffusion. (External Knowledge: 'a few' refers to vector similarity search with k=3 limit)''',

"Metaphorical": '''**Metaphorical Style**
   - Uses metaphors or metaphorical expressions.
   - Example: Find the names and addresses of those who have reached adulthood. (External Knowledge: 'reached adulthood' refers to age >= 18.)
   - Vector Example: Find a few articles have to do with SD in ai. (External Knowledge: 'SD in ai' refers to Stable Diffusion)''',

"Multi-turn Dialogue": '''**Multi-turn Dialogue Style**
    - This involves a dialogue to clarify the user's query needs.
    - Example: [{"User": "I want to query some student information."}, {"Assistant": "Which students' information would you like to query?"}, {"User": "Students older than 18."}, {"Assistant": "What other information would you like to know about them?"}, {"User": "Names and addresses."}, {"Assistant": "Is there anything else you need?"}, {"User": "No."}, {"Assistant": "OK, I will help you translate your request into an SQL query."}]
    - Vector Example: 
      User: "I'm looking for some articles."
      Assistant: "How many articles would you like to find and What field of paper are you looking for?"
      User: "About 3, and they are related to Stable Diffusion."
      Assistant: "I'll search for 3 articles that most closely related to Stable Diffusion."'''
}

# 步骤说明 (无需改动)
steps_wo_ek = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does, including any vector search operations.
2. **Generate a Question:** Formulate a natural language question based on the SQL query and explanation.'''

steps_w_ek = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does, including any vector search operations.
2. **Generate a Question:** Formulate a natural language question based on the SQL query and explanation.
3. **External Knowledge:** For Vague or Metaphorical styles, include external knowledge to enhance clarity, especially for vector operations.'''

steps_multi_round = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does, including any vector search operations.
2. **Generate a Dialogue:** Create a conversation between the User and the Assistant based on the SQL query and its explanation, ensuring vector operations are properly discussed.'''

# 指导方针 (无需改动)
guidelines_wo_ek = '''1. Clearly describe the columns being selected by the SQL query. For example:
   - "SELECT * ... FROM ..." means "Find all ...";
   - "SELECT f.check_date, f.status, f.remarks, c.year, c.year_min, c.year_max, c.year_average, c.data_quality_score FROM ..." means "Return the check dates, statuses, remarks, years, minimum years, maximum years, average years, and quality scores for ...".
   - "SELECT rowid, vec FROM vec_table WHERE vec MATCH lembed(_,"xxx") ORDER BY distance LIMIT 2;" means "Return two of the rowid and vec that most related to xxx from vec_table, ordered by similarity distance".
   - "SELECT rowid, vec FROM vec_table WHERE vec MATCH lembed(_,"xxx") AND k = 2;" means "Return two of the rowid and vec that most related to xxx from vec_table, ordered by similarity distance".
   - For vector searches: Always mention the LIMIT value or K value when explaining MATCH operations.

2. Ensure the natural language question accurately captures:
   - All conditions including vector similarity searches
   - ORDER BY clauses (especially for distance/similarity)
   - LIMIT and K clauses
   - Any window functions or complex joins'''

guidelines_w_ek = '''1. Clearly describe the columns being selected by the SQL query (same as above).
2. Ensure the natural language question captures all query semantics (same as above).
3. For vector searches, include these common external knowledge points:
   - "MATCH" operator performs approximate nearest neighbor (ANN) search;
   - "k=N" specifies the number of similar items to return;
   - Vectors are compared using Euclidean distance (L2 norm) by default;
   - Similarity increases as distance decreases;
   - Include any domain-specific knowledge about the vector meaning.'''

guidelines_multi_round = '''1. Clearly describe the columns being selected by the SQL query (same as above).
2. Ensure the dialogue naturally covers:
   - The purpose of the vector search;
   - How many similar items are needed (LIMIT);
   - What the target vector represents;
   - Any additional filtering or sorting requirements.'''


# ##########################################################################
# # MODIFIED: This entire block is updated with more nuanced instructions. #
# ##########################################################################
vector_question_guidelines = '''

**Crucial Rule for Vector Queries**: Your most important task is to convert the technical `lembed()` search text into a natural, human-like question that matches the specified style, while preserving the core meaning.

1.  **Identify and Preserve Key Entities**: First, identify the core concepts and keywords within the `lembed()` text. For example, in `"Professor of Computer Science"`, the key entities are **"Professor"** and **"Computer Science"**. These key entities **MUST BE PRESERVED** in the generated question to maintain its meaning.
2.  **Rephrase Naturally, Do Not Copy Verbatim**: Your goal is to rephrase the overall sentence structure to sound natural. **Do not copy the entire `lembed()` string verbatim**, but you should use the key entities you identified. For example, instead of copying `"Professor of Computer Science"`, you could rephrase it as `"a computer science professor"` or `"professors who specialize in computer science"`. The phrasing must match the requested style (e.g., Formal, Colloquial).
3.  **Focus on User Intent**: The final question should sound like a real user asking for information, not like a description of a technical process.

---
**Examples of Correct vs. Incorrect Behavior:**

**Example 1: Preserving Entities in "Formal" Style**
* **Input VecSQL**: `... WHERE p.hasPosition_embedding MATCH lembed('all-MiniLM-L6-v2', "Professor of Computer Science") AND k = 5 ...`
* **BAD Question**: `"Identify five professors whose roles most closely match the concept of teaching computer science at a professorial level..."`
    * **Reasoning**: Too verbose and abstract. "concept of teaching computer science at a professorial level" is an unnatural way to say "Computer Science Professor".
* **GOOD Question**: `"Please provide the IDs and course levels for the 5 individuals most similar to a 'Professor of Computer Science'."` or `"Identify five professors specializing in Computer Science and list the levels of the courses they teach."`
    * **Reasoning**: Correctly uses the key entities "Professor" and "Computer Science" in a formal and direct manner.

**Example 2: Avoiding Over-generalization in "Imperative" Style**
* **Input VecSQL**: `... WHERE p.hasPosition_embedding MATCH lembed('all-MiniLM-L6-v2', "Professor of Mathematics") LIMIT 5 ...`
* **BAD Question**: `"Could you please find the top 5 individuals most semantically related to a specialized academic teaching role..."`
    * **Reasoning**: Completely fails by losing the essential key entities **"Professor"** and **"Mathematics"**. The question is now uselessly vague.
* **GOOD Question**: `"Get me the top 5 people who are most like a Professor of Mathematics, and show me their course levels."`
    * **Reasoning**: Preserves the critical entities in a natural, imperative sentence.

**Example 3: Being Natural in "Colloquial" Style**
* **Input VecSQL**: `... WHERE performance_embedding MATCH lembed('all-MiniLM-L6-v2', "exceptional performance with leadership skills") LIMIT 1;`
* **BAD Question**: `"Hey, could you help me find the employee whose performance is most closely related to being a standout leader?"`
    * **Reasoning**: "most closely related to being..." is clunky. It also loses the "exceptional performance" aspect.
* **GOOD Question**: `"Hey, can you find the employee who is the best example of 'exceptional performance with leadership skills'? I need their SSN."` or `"Who's our top employee showing both great performance and leadership? Grab their SSN for me."`
    * **Reasoning**: Sounds like a real person talking and naturally incorporates the key concepts.
---'''
# ##########################################################################
# # End of Modified Block                                                  #
# ##########################################################################


# 输出格式 (无需改动)
output_format_wo_ek = '''Please structure your response as follows:

[EXPLANATION-START]
(SQL Explanation including vector operations if present)
[EXPLANATION-END]

[QUESTION-START]
(Natural Language Question capturing all query elements)
[QUESTION-END]'''

output_format_w_ek = '''Please structure your response as follows:

[EXPLANATION-START]
(SQL Explanation including vector operations)
[EXPLANATION-END]

[QUESTION-START]
(Natural Language Question)
[QUESTION-END]

[EXTERNAL-KNOWLEDGE-START]
(Relevant knowledge about vector operations and domain context)
[EXTERNAL-KNOWLEDGE-END]'''

output_format_multi_round = '''Please structure your response as follows:

[EXPLANATION-START]
(SQL Explanation including vector operations)
[EXPLANATION-END]

[QUESTION-START]
(Multi-turn dialogue covering vector search details)
[QUESTION-END]'''

# 指令 (无需改动)
instruction_wo_ek = '''Based on the above information:
1. Analyze the SQL query, paying special attention to any vector operations
2. Generate a clear explanation covering all query elements
3. Formulate a precise natural language question
4. Verify all vector operations (MATCH, LIMIT, ORDER BY distance) or (MATCH, And k = ?) are properly represented'''

instruction_w_ek = '''Based on the above information:
1. Analyze the SQL query, especially vector operations
2. Generate explanation covering all elements
3. Formulate precise question
4. Add relevant external knowledge about vector operations
5. Verify all vector elements are properly represented'''

instruction_multi_round = '''Based on the above information:
1. Analyze the SQL query, especially vector operations
2. Generate explanation covering all elements
3. Create natural dialogue that explores vector search parameters
4. Ensure LIMIT, target vector and distance sorting are discussed'''


def contains_virtual_table(statements):
    if isinstance(statements, str):
        statements = [statements]
    patterns = [
        r'\bvirtual\b', r'\bvec0\b', r'_embedding\b', r'\bfloat\[', r'\]\b'
    ]
    for stmt in statements:
        if not stmt:
            continue
        for pattern in patterns:
            if re.search(pattern, stmt, re.IGNORECASE):
                return True
    return False

def obtain_db_schema(db_file_dir):
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn) 
    sqlite_lembed.load(conn)
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]
    create_statements = [table[1] for table in tables]
    cursor.close()
    conn.close()
    return table_names, create_statements

def extract_column_descriptions(create_statements):
    column_name2column_desc = dict()
    pattern = r'"(\w+)"\s+(\w+)\s*/\*\s*(.*?)\s*\*/'
    for create_statement in create_statements:
        matches = re.findall(pattern, create_statement)
        for column_name, col_type, description in matches:
            column_name = column_name.lower()
            if col_type.upper() in ('VECTOR', 'FLOAT[]', 'FLOAT[%d]'):
                desc = f"Vector column: {description}" if description else "Floating-point vector column"
            else:
                desc = description if description else f"{col_type} column"
            column_name2column_desc[column_name] = desc
    return column_name2column_desc

def detect_vector_columns(sql):
    pattern = r'MATCH\s+(?:lembed\([^)]+\)|\'\[[^\]]+\]\')'
    return bool(re.search(pattern, sql, re.IGNORECASE))

def enhance_vector_info(sql, column_info):
    if detect_vector_columns(sql):
        for col in list(column_info.keys()):
            if '_embedding' in col:
                column_info[col] = f"Vector column for similarity search: {column_info.get(col, '')}"
    return column_info

if __name__ == "__main__":
    random.seed(42)
    db_path = "../brid_vectorization/results/vector_databases_brid"
    sql_infos = json.load(open("../sql_synthesis/results/synthetic_sqls.json"))
    question_synthesis_template = open("./prompt_templates/question_synthesis_prompt.txt").read()
    styles = list(style2desc.keys())

    os.makedirs("./prompts", exist_ok=True)

    db_ids = list(set([sql["db_id"] for sql in sql_infos]))
    db_id2column_info = dict()
    db_id_vec_flag = dict()
    
    for db_id in tqdm(db_ids, desc="Processing databases"):
        table_names, create_statements = obtain_db_schema(os.path.join(db_path, db_id, db_id + ".sqlite"))
        db_id_vec_flag[db_id] = contains_virtual_table(create_statements)
        db_id2column_info[db_id] = extract_column_descriptions(create_statements)
    
    prompts = []
    for sql_info in tqdm(sql_infos, desc="Generating prompts"):
        style_name = random.choice(styles)
        column_info = db_id2column_info[sql_info["db_id"]]
        
        column_info = enhance_vector_info(sql_info["sql"], column_info)
        
        used_columns = {}
        sql_lower = sql_info["sql"].lower()
        for col_name, col_desc in column_info.items():
            if col_name.lower() in sql_lower:
                used_columns[col_name] = col_desc

        if style_name in ["Vague", "Metaphorical"]:
            steps, guidelines, instruction, output_format = steps_w_ek, guidelines_w_ek, instruction_w_ek, output_format_w_ek
        elif style_name == "Multi-turn Dialogue":
            steps, guidelines, instruction, output_format = steps_multi_round, guidelines_multi_round, instruction_multi_round, output_format_multi_round
        else:
            steps, guidelines, instruction, output_format = steps_wo_ek, guidelines_wo_ek, instruction_wo_ek, output_format_wo_ek
        
        is_vector_query = detect_vector_columns(sql_info["sql"])
        if is_vector_query:
            guidelines += vector_question_guidelines
            
        prompt_with_vector = ""
        if db_id_vec_flag.get(sql_info["db_id"], False):
            prompt_with_vector = "You have to use KNN queries, if extension includes sqlite-vec."
        
        prompt = question_synthesis_template.format(
            using_knn=prompt_with_vector,
            style_desc=style2desc[style_name].strip(),
            engine="SQLite",
            extension="sqlite-vec and sqlite-lembed",
            column_info=json.dumps(used_columns, indent=2, ensure_ascii=False).strip(),
            sql=sql_info["sql"].strip(),
            steps=steps.strip(),
            guidelines=guidelines.strip(),
            output_format=output_format.strip(),
            instruction=instruction.strip()
        )
        
        sql_info["style"] = style_name
        sql_info["prompt"] = prompt
        sql_info["contains_vector"] = is_vector_query
    
    with open("./prompts/question_synthesis_prompts.json", "w", encoding="utf-8") as f:
        json.dump(sql_infos, f, indent=2, ensure_ascii=False)
