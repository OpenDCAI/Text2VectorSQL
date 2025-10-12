
import os
import re
import json
import random
import sqlite3
import traceback

def merge_foreign_keys_to_create_table(create_stmts, fk_stmts):
    # Extract foreign key constraint information
    #ALTER TABLE "performance_metrics" ADD CONSTRAINT fk_performance_metrics_app_id FOREIGN KEY ("app_id") REFERENCES applications ("app_id");
    fk_constraints = {}
    for alter_statement in fk_stmts:
        match = re.search(r'ALTER TABLE "(\w+)" ADD CONSTRAINT (\w+) FOREIGN KEY \("(\w+)"\) REFERENCES (\w+) \("(\w+)"\)', alter_statement)
        if match:
            table_name = match.group(1)
            constraint_name = match.group(2)
            column_name = match.group(3)
            ref_table_name = match.group(4)
            ref_column_name = match.group(5)
            if table_name in fk_constraints:
                fk_constraints[table_name].append(f'CONSTRAINT {constraint_name} FOREIGN KEY ("{column_name}") REFERENCES {ref_table_name} ("{ref_column_name}")')
            else:
                fk_constraints[table_name] = [f'CONSTRAINT {constraint_name} FOREIGN KEY ("{column_name}") REFERENCES {ref_table_name} ("{ref_column_name}")']

    # Merge foreign key constraints into the CREATE TABLE statement
    modified_create_table_statements = []
    for create_statement in create_stmts:
        match = re.search(r'CREATE TABLE "(\w+)"', create_statement)
        if match:
            table_name = match.group(1)
            if table_name in fk_constraints:
                for fk in fk_constraints[table_name]:
                    create_statement = create_statement.rstrip('\n);') + '), \n  ' + fk + '\n);'
        modified_create_table_statements.append(create_statement)
    return modified_create_table_statements

def verify_ddl_in_transaction(ddl_stmts, db_id):
    create_stmts = ddl_stmts['create_stmts']
    insert_stmts = ddl_stmts['insert_stmts']
    alter_stmts = ddl_stmts['alter_stmts']
    fk_stmts = ddl_stmts['fk_stmts']
    stmts = merge_foreign_keys_to_create_table(create_stmts, fk_stmts)
    
    os.makedirs(f'synthesis_data/{db_id}', exist_ok=True)
    try:
        # connect db
        conn = sqlite3.connect(f'synthesis_data/{db_id}/{db_id}.sqlite')
        cursor = conn.cursor()

        # begin transaction
        conn.execute('BEGIN TRANSACTION')
        cursor.execute('PRAGMA foreign_keys = OFF;')

        # CREATE TABLE
        for stmt in stmts:
            # print(stmt)
            try:
                cursor.execute(stmt)
            except Exception as e:
                # print("Exception: ", str(e))
                continue
        
        # INSERT INTO
        for stmt in insert_stmts:
            # print(stmt)
            try:
                cursor.execute(stmt) 
            except Exception as e:
                # print("Exception: ", str(e))
                continue       
        
        cursor.execute('PRAGMA foreign_keys = ON;')

        # update values in foreign key columns
        for alter_stmt in alter_stmts:
            stmt = alter_stmt['alter_stmt']
            values = alter_stmt['values']
                
            # create an empty dict to fill placeholder
            filled_values = {}
            for i, value in enumerate(values):
                tp = value['type']
                rg = value['range']
                v = random.randint(0, rg)
                if tp == "TEXT":
                    v = str(v)
                elif tp == "INTEGER":
                    v = int(v)
                filled_values[f'id_{i}'] = v
                
            stmt = stmt.format(**filled_values)
            try:
                cursor.execute(stmt)
            except Exception as e:
                # print("Exception: ", str(e))
                continue


        # commit transaciton
        conn.commit()
        print("Transaction committed successfully.")

    except Exception as e:
        # if any error occurs, roll back the transaction
        conn.rollback()
        print("Transaction failed and rolled back. Error:", str(e))
        raise Exception()
    finally:
        # close the connection
        conn.close()

def convert_complex_type(sql_type):
    """Converts complex types such as Array and Struct to SQLite-compatible types."""
    if "Array" in sql_type:
        return "TEXT"  # Convert Array to TEXT (as JSON-encoded strings)
    elif "Struct" in sql_type:
        return "TEXT"  # Convert Struct to TEXT (as JSON-encoded strings)
    else:
        # Mapping for standard types
        type_mapping = {
            "INTEGER": "INTEGER",
            "VARCHAR": "TEXT",  # SQLite treats all VARCHAR as TEXT
            "TEXT": "TEXT",
            "REAL": "REAL",
            "FLOAT": "REAL",
            "DATE": "TEXT",
            "TIME": "TEXT",
            "BOOLEAN": "INTEGER"  # SQLite uses INTEGER for boolean
        }
        return type_mapping.get(sql_type, "TEXT")  # Default to TEXT if unknown type


def format_value_for_sqlite(value, column_type):
    """Formats values for SQLite, including handling Array and Struct types."""
    if "Array" in column_type or "Struct" in column_type:
        # Convert complex types (Array, Struct) to JSON strings
        return f"'{json.dumps(value)}'"
    elif isinstance(value, str):
        # Escape single quotes in strings using replace before f-string
        value = value.replace("'", "''")
        return f"'{value}'"
    elif value is None:
        return "NULL"
    return str(value)


def generate_sqlite_ddl(json_schema):
    """Generates SQLite DDL statements including primary and foreign keys, table descriptions, and sample row insertion."""
    result = {}
    ddl_statements = []
    insert_stmts = []
    foreign_key_statements = set()
    foreign_keys_alter = {}
    foreign_keys_alter_stmts = []
    rows_cnt = {}
    table_pk = {}
    table_cols = {}
    table_types = {}
    
    for table in json_schema['tables']:
        table_name = table['table_name']
        table_description = table.get('table_description', '')
        column_names = table['column_names']
        column_types = table['column_types']
        descriptions = table['column_descriptions']
        primary_key = table.get('primary_key', [])
        sample_rows = table.get('sample_rows', [])

        # Step 1: Create table comment (table description as a comment)
        # if table_description:
        #     ddl_statements.append(f'-- {table_description}')

        # Step 2: Create table without foreign key constraints
        columns_ddl = []
        table_cols[table_name] = column_names
        table_types[table_name] = column_types
        for i, column_name in enumerate(column_names):
            column_type = convert_complex_type(column_types[i])
            description = descriptions[i]
            columns_ddl.append(f'"{column_name}" {column_type} /* {description} */')

        # Add primary key constraint
        if primary_key:
            table_pk[table_name] = primary_key
            pk_columns = ', '.join(f'"{col}"' for col in primary_key)
            columns_ddl.append(f'PRIMARY KEY ({pk_columns})')

        ddl = f'CREATE TABLE "{table_name}" (\n  ' + ',\n  '.join(columns_ddl) + '\n);'
        ddl_statements.append(ddl)
    
        rows_cnt[table_name] = len(sample_rows)

        # Insert sample rows
        if sample_rows:
            for idx, row in enumerate(sample_rows):
                # if idx > 2: # 
                #     break
                # Find the index of the primary key column
                pk_indices = [column_names.index(key) for key in primary_key]
                values = [format_value_for_sqlite(value, column_types[i]) for i, value in enumerate(row)]
                for pk_idx in pk_indices:
                    type_str = convert_complex_type(column_types[pk_idx])
                    if type_str == 'TEXT':
                        values[pk_idx] = str(idx)
                    elif type_str == 'INTEGER':
                        values[pk_idx] = idx
                    elif type_str == "REAL":
                        values[pk_idx] = float(idx)
                
                if len(column_names) != len(values):
                    continue
                values = ", ".join([str(value) for value in values])
                # print(values)
                insert_stmt = f'INSERT INTO "{table_name}" ({", ".join(column_names)}) VALUES ({values});'
                insert_stmts.append(insert_stmt)
    
    table_sets = {}
    for table_name, pks in table_pk.items():
        table_sets[table_name] = set(pks)
    for fk in json_schema['foreign_keys']:
        table_name = fk['source_table']
        src_cols = fk['column_in_source_table'] if type(fk['column_in_source_table']) == list else [fk['column_in_source_table']]
        ref_cols = fk['column_in_referenced_table'] if type(fk['column_in_referenced_table']) == list else [fk['column_in_referenced_table']]
        real_src_cols = []
        real_ref_cols = []
        for src_col, ref_col in zip(src_cols, ref_cols):
            if src_col in table_sets[table_name]:
                continue
            real_ref_cols.append(ref_col)
            real_src_cols.append(src_col)
        if len(real_src_cols) == 0:
            continue
        fk_source_cols = ', '.join(f'"{col}"' for col in real_src_cols)
        fk_ref_table = fk['referenced_table']
        fk_ref_cols = ', '.join(f'"{col}"' for col in real_ref_cols)
        column_names = table_cols[table_name]
        column_types = table_types[table_name]
        fk_stmt = (f'ALTER TABLE "{table_name}" '
                    f'ADD CONSTRAINT fk_{table_name}_{"_".join(real_src_cols)} '
                   f'FOREIGN KEY ({fk_source_cols}) REFERENCES {fk_ref_table} ({fk_ref_cols});')
        if fk_stmt in foreign_key_statements:
            continue
        foreign_key_statements.add(fk_stmt)
        if table_name in foreign_keys_alter:
            for i in range(len(real_src_cols)):
                foreign_keys_alter[table_name]['ref_table'].append(fk_ref_table)
            foreign_keys_alter[table_name]['fk_cols'].extend(real_src_cols)
            foreign_keys_alter[table_name]['fk_types'].extend([convert_complex_type(column_types[column_names.index(fk)]) for fk in real_src_cols])
        else:
            foreign_keys_alter[table_name] = {
                "src_table": table_name,
                "ref_table": [fk_ref_table],
                "fk_cols": real_src_cols,
                "fk_types": [convert_complex_type(column_types[column_names.index(fk)]) for fk in real_src_cols],
                "pk_cols": table_pk[table_name],
                "pk_types": [convert_complex_type(column_types[column_names.index(pk)]) for pk in table_pk[table_name]]
            }
                
    # for stmt in ddl_statements:
    #     pass
    # Alter table for foreign key constraint DDL
    for table_name, fk_alter in foreign_keys_alter.items():
        source_table = fk_alter["src_table"]
        ref_table = fk_alter["ref_table"]
        src_row_num = rows_cnt[source_table]
        ref_row_num = [rows_cnt[ref] for ref in ref_table]
        pk_cols = fk_alter["pk_cols"]
        pk_types = fk_alter["pk_types"]
        cols = fk_alter["fk_cols"]
        types = fk_alter["fk_types"]
        for i in range(src_row_num):
            ddl_stmt = f"UPDATE {source_table} SET "
            fk_des = []
            for j, col, tp in zip(range(len(cols)), cols, types):
                id = random.randint(0, ref_row_num[j]-1)
                fk_des.append({"type": tp, "range": ref_row_num[j]-1})
                if tp == "TEXT":
                    id = str(id)
                elif tp == "REAL":
                    id = float(id)
                ddl_stmt += (f"{col}"+" = {id_"+str(j)+"}, ")
            ddl_stmt = ddl_stmt.strip()[:-1] + " WHERE "
            for j, pk, ptp in zip(range(len(pk_cols)) , pk_cols, pk_types):
                i_v = i
                if ptp == "TEXT":
                    i_v = str(i_v)
                elif ptp == "REAL":
                    i_v = float(i_v)
                if j == 0:
                    ddl_stmt += f"{pk} = {i_v}"
                else:
                    ddl_stmt += f" and {pk} = {i_v}"
            ddl_stmt += ";"
            foreign_keys_alter_stmts.append({"alter_stmt": ddl_stmt, "values": fk_des})
                # execute update
    
    # for stmt in foreign_key_statements:
    #     pass
    result["create_stmts"] = ddl_statements
    result["insert_stmts"] = insert_stmts
    result["alter_stmts"] = foreign_keys_alter_stmts
    result["fk_stmts"] = list(foreign_key_statements)

    return result


# Example usage:
json_schema_str = '''{
  "tables": [
    {
      "table_name": "datasets",
      "table_description": "Stores details of all greenhouse gas datasets collected from global sites.",
      "column_names": ["dataset_id", "dataset_number", "site_id", "category", "gas_name", "sampling_method", "frequency", "year", "download_link", "readme_link"],
      "column_types": ["INTEGER", "INTEGER", "INTEGER", "VARCHAR", "VARCHAR", "VARCHAR", "VARCHAR", "INTEGER", "VARCHAR", "VARCHAR"],
      "column_descriptions": [
        "Unique identifier for each dataset",
        "Number assigned to the dataset",
        "Reference to the site where the data was collected",
        "Category of the data (e.g., Greenhouse Gases)",
        "Name of the gas being monitored",
        "Method of sampling (e.g., Surface PFP, Aircraft PFP, Flask)",
        "Sampling frequency (e.g., Discrete, Continuous)",
        "Year when the data was collected",
        "Link to download the dataset",
        "Link to the readme or metadata of the dataset"
      ],
      "primary_key": ["dataset_id"],
      "sample_rows": [
        [151, 151, 1, "Greenhouse Gases", "Carbon Dioxide(CO2)", "Surface PFP", "Discrete", 2023, "download_link_151", "readme_link_151"],
        [152, 152, 2, "Greenhouse Gases", "Carbon Dioxide(CO2)", "Aircraft PFP", "Discrete", 2023, "download_link_152", "readme_link_152"]
      ]
    },
    {
      "table_name": "sites",
      "table_description": "Details of the sites where greenhouse gas samples are collected.",
      "column_names": ["site_id", "site_name", "location", "country", "contact_email"],
      "column_types": ["INTEGER", "VARCHAR", "VARCHAR", "VARCHAR", "VARCHAR"],
      "column_descriptions": [
        "Unique identifier for each site",
        "Name of the site",
        "Geographical location of the site",
        "Country where the site is located",
        "Contact email for the site or environmental team"
      ],
      "primary_key": ["site_id"],
      "sample_rows": [
        [1, "West Branch, Iowa", "West Branch, Iowa, United States", "USA", "contact@westbranch.us"],
        [2, "Walnut Grove, California", "Walnut Grove, California, United States", "USA", "contact@walnutgrove.us"]
      ]
    },
    {
      "table_name": "sampling_methods",
      "table_description": "Details of various sampling methods used for collecting air samples.",
      "column_names": ["method_id", "method_name", "description"],
      "column_types": ["INTEGER", "VARCHAR", "TEXT"],
      "column_descriptions": [
        "Unique identifier for each sampling method",
        "Name of the sampling method (e.g., Surface PFP, Aircraft PFP)",
        "Detailed description of the sampling method"
      ],
      "primary_key": ["method_id"],
      "sample_rows": [
        [1, "Surface PFP", "Surface flask sampling for air composition"],
        [2, "Aircraft PFP", "Aircraft-based flask sampling for higher altitude air"]
      ]
    },
    {
      "table_name": "gas_samples",
      "table_description": "Raw data of the gas concentrations measured at each site.",
      "column_names": ["sample_id", "dataset_id", "gas_name", "concentration", "measurement_date", "measurement_time"],
      "column_types": ["INTEGER", "INTEGER", "VARCHAR", "FLOAT", "DATE", "TIME"],
      "column_descriptions": [
        "Unique identifier for each gas sample",
        "Reference to the dataset from which the sample is drawn",
        "Name of the gas measured (e.g., CO2, CH4)",
        "Concentration of the gas in ppm (parts per million)",
        "Date of the measurement",
        "Time of the measurement"
      ],
      "primary_key": ["sample_id"],
      "sample_rows": [
        [1, 151, "Carbon Dioxide(CO2)", 405.2, "2023-05-01", "12:00:00"],
        [2, 152, "Carbon Dioxide(CO2)", 407.8, "2023-05-02", "12:30:00"]
      ]
    },
    {
      "table_name": "users",
      "table_description": "Details of users accessing the datasets and samples.",
      "column_names": ["user_id", "user_name", "email", "organization", "role"],
      "column_types": ["INTEGER", "VARCHAR", "VARCHAR", "VARCHAR", "VARCHAR"],
      "column_descriptions": [
        "Unique identifier for each user",
        "Full name of the user",
        "Email address of the user",
        "Organization the user belongs to",
        "Role of the user (e.g., researcher, admin, viewer)"
      ],
      "primary_key": ["user_id"],
      "sample_rows": [
        [101, "Dr. Alice Green", "alice.green@enviroresearch.org", "EnviroResearch", "researcher"],
        [102, "John Doe", "john.doe@climatelabs.org", "Climate Labs", "admin"]
      ]
    }
  ],
  "foreign_keys": [
    {
      "source_table": "datasets",
      "column_in_source_table": "site_id",
      "referenced_table": "sites",
      "column_in_referenced_table": "site_id"
    },
    {
      "source_table": "gas_samples",
      "column_in_source_table": "dataset_id",
      "referenced_table": "datasets",
      "column_in_referenced_table": "dataset_id"
    }
  ]
}'''

def verify_schema(json_schema, db_id):
    # Convert the schema into DDL statements
    try:
        ddl_stmts = generate_sqlite_ddl(json_schema)
        verify_ddl_in_transaction(ddl_stmts, db_id)
        return True
    except Exception as e:
        print("Exception type:", type(e))
        print("Exception message:", e)
        # traceback.print_exc()
        return False

    # Print the DDL output
    # print(ddl_output["create_stmts"])
    # print(ddl_output["insert_stmts"])
    # print(ddl_output["alter_stmts"])
    # print(ddl_output["fk_stmts"])

if __name__ == "__main__":
    verify_schema(json.loads(json_schema_str), "test_db")
