import json
from tqdm import tqdm
from sqlite_schema_parser import verify_schema
import random

if __name__ == "__main__":
    enhanced_results = json.load(open("./results/schema_enhancement.json"))

    final_schemas = []
    error_case_num = 0
    for result in tqdm(enhanced_results):
        try:
            domain = result["domain"]
            schema = json.loads(result["enhanced_schema"])
            assert "tables" in schema and "foreign_keys" in schema

            tables = []
            for table in schema["tables"]:
                try:
                    assert "table_name" in table and "column_names" in table and \
                        "column_types" in table and "column_descriptions" in table
                    assert len(table["column_names"]) == len(table["column_types"]) == len(table["column_descriptions"])
                    tables.append(table)
                except Exception as e:
                    pass

            table_names_lower = [table["table_name"].lower() for table in tables]
            
            foreign_keys = []
            for foreign_key in schema["foreign_keys"]:
                try:
                    assert "source_table" in foreign_key and "column_in_source_table" in foreign_key and \
                        "referenced_table" in foreign_key and "column_in_referenced_table" in foreign_key
                    assert foreign_key["source_table"].lower() in table_names_lower and \
                        foreign_key["referenced_table"].lower() in table_names_lower
                    foreign_keys.append(foreign_key)
                except Exception as e:
                    pass
            
            final_schemas.append(
                {
                    "domain": domain,
                    "tables": tables,
                    "foreign_keys": foreign_keys
                }
            )
        except Exception as e:
            error_case_num += 1
            # print(e)
    print("error_case_num:", error_case_num)

    db_ids = []
    success_labels = []
    for final_schema in tqdm(final_schemas):
        db_id = final_schema["domain"].lower().replace("(", "_").replace(")", "_").replace("-", "_").replace(" ", "_").replace("*", "_").strip()

        if len(db_id) > 75:
            db_id = db_id[:75]
        
        # resolve db_id conflict issues
        while db_id in db_ids:
            db_id += "_" + str(random.randint(0, 1000000000000))

        success_label = verify_schema(final_schema, db_id)
        if success_label:
            db_ids.append(db_id)
    
        success_labels.append(success_label)
    
    print("success rate:", sum(success_labels)/len(success_labels))
