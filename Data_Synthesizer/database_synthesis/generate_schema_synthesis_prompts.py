import json
import random
import numpy as np
import os
# random.seed(42)

def generate_a_normal_integer(mean = 10, std_dev = 4, lower_bound = 1, upper_bound = 20):
    sample = np.random.normal(mean, std_dev)
    sample = np.clip(sample, lower_bound, upper_bound)
    
    return int(sample)

def generate_schema_form():
    sample = random.randint(0, 4)
    if sample == 0:
        return ("All table and column names must be capitalized.",
        """
{{
  "table_num": 5,
  "tables": [
    {{
      "table_name": "DOCUMENTS",
      "table_description": "Table to store details of each document uploaded to the platform.",
      "column_names": [
        "DOC_ID",
        "DOC_TITLE",
        "DOC_VERSION",
        "SITE_ID",
        "UPLOAD_DATE",
        "LAST_UPDATED",
        "HITS",
        "AUTHOR_ID"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "INTEGER",
        "INTEGER",
        "DATE",
        "DATE",
        "INTEGER",
        "INTEGER"
      ],
      "column_descriptions": [
        "Unique identifier for each document",
        "Title of the document",
        "Version number of the document",
        "Reference to the site or department where the document is hosted",
        "Date the document was uploaded",
        "Date the document was last updated",
        "Number of views or hits the document has received",
        "ID of the author who uploaded the document"
      ],
      "primary_key": [
        "DOC_ID"
      ],
      "sample_rows": [
        [
          120983,
          "Canvas, How to Change the Course Name",
          1,
          101,
          "2022-08-01",
          "2022-08-31",
          821,
          3001
        ],
        [
          120334,
          "Contracts+ Using Jaggaer Word App",
          1,
          102,
          "203-04-01",
          "2023-04-24",
          822,
          3002
        ]
      ]
    }},
    {{
      "table_name": "SITES",
      "table_description": "Details of the sites or departments hosting the documents.",
      "column_names": [
        "SITE_ID",
        "SITE_NAME",
        "DEPARTMENT",
        "CONTACT_EMAIL"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each site",
        "Name of the site or department",
        "Department under which the site operates",
        "Contact email for inquiries about the site"
      ],
      "primary_key": [
        "SITE_ID"
      ],
      "sample_rows": [
        [
          101,
          "University of Illinois Technology Services",
          "Technology Services",
          "techsupport@uillinois.edu"
        ],
        [
          102,
          "UI Training and Development Resources",
          "Training and Development",
          "train@uillinois.edu"
        ]
      ]
    }},
    {{
      "table_name": "AUTHORS",
      "table_description": "Information about the authors uploading the documents.",
      "column_names": [
        "AUTHORID",
        "AUTHORNAME",
        "EMAIL",
        "DEPARTMENT"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each author",
        "Full name of the author",
        "Email address of the author",
        "Department the author belongs to"
      ],
      "primary_key": [
        "AUTHOR_ID"
      ],
      "sample_rows": [
        [
          3001,
          "John Doe",
          "jdoe@uillinois.edu",
          "Technology Services"
        ],
        [
          3002,
          "Jane Smith",
          "jsmith@uillinois.edu",
          "Training and Development"
        ]
      ]
    }},
    {{
      "table_name": "DOCUMENT_ACCESS",
      "table_description": "Tracks access to documents by users.",
      "column_names": [
        "ACCESS_ID",
        "DOC_ID",
        "USER_ID",
        "ACCESS_DATE",
        "ACCESS_TYPE"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "INTEGER",
        "DATE",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each access event",
        "ID of the document being accessed",
        "ID of the user accessing the document",
        "Date when the document was accessed",
        "Type of access (e.g., view, download)"
      ],
      "primary_key": [
        "ACCESS_ID"
      ],
      "sample_rows": [
        [
          5001,
          120983,
          4001,
          "2023-05-01",
          "view"
        ],
        [
          5002,
          120334,
          4002,
          "2023-05-02",
          "download"
        ]
      ]
    }},
    {{
      "table_name": "USERS",
      "table_description": "Details of users accessing the documents.",
      "column_names": [
        "USER_ID",
        "USER_NAME",
        "EMAIL",
        "ROLE"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each user",
        "Full name of the user",
        "Email address of the user",
        "Role of the user (e.g., student, staff, admin)"
      ],
      "primary_key": [
        "USER_ID"
      ],
      "sample_rows": [
        [
          4001,
          "Alice Johnson",
          "alice.johnson@uillinois.edu",
          "student"
        ],
        [
          4002,
          "Bob Williams",
          "bob.williams@uillinois.edu",
          "staff"
        ]
      ]
    }}
  ],
  "foreign_keys": [
    {{
      "source_table": "DOCUMENTS",
      "column_in_source_table": "SITE_ID",
      "referenced_table": "SITES",
      "column_in_referenced_table": "SITE_ID"
    }},
    {{
      "source_table": "DOCUMENTS",
      "column_in_source_table": "AUTHOR_ID",
      "referenced_table": "AUTHORS",
      "column_in_referenced_table": "AUTHORID"
    }},
    {{
      "source_table": "DOCUMENT_ACCESS",
      "column_in_source_table": "DOC_ID",
      "referenced_table": "DOCUMENTS",
      "column_in_referenced_table": "DOC_ID"
    }},
    {{
      "source_table": "DOCUMENT_ACCESS",
      "column_in_source_table": "USER_ID",
      "referenced_table": "USERS",
      "column_in_referenced_table": "USER_ID"
    }}
  ]
}}        
        """,
        """
{{
  "table_num": 10,
  "tables": [
    {{
      "table_name": "DATASETS",
      "table_description": "Table to store details of each dataset collected from various sites.",
      "column_names": [
        "DATASET_ID",
        "SITE_ID",
        "CATEGORY",
        "NAME",
        "TYPE",
        "FREQUENCY",
        "YEAR",
        "DATA_FILE",
        "README_FILE"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each dataset",
        "Reference to the site where the data was collected",
        "Category of the data (e.g., Greenhouse Gases)",
        "Name of the data (e.g., Carbon Dioxide(CO2))",
        "Type of data collection method (e.g., Surface PFP, Aircraft PFP)",
        "Frequency of data collection (e.g., Discrete, Continuous)",
        "Year(s) the data was collected",
        "File path to the data file",
        "File path to the readme file"
      ],
      "primary_key": [
        "DATASET_ID"
      ],
      "sample_rows": [
        [
          151,
          1001,
          "Greenhouse Gases",
          "Carbon Dioxide(CO2)",
          "Surface PFP",
          "Discrete",
          "Multiple",
          "data/151.csv",
          "readme/151.txt"
        ],
        [
          152,
          1002,
          "Greenhouse Gases",
          "Carbon Dioxide(CO2)",
          "Aircraft PFP",
          "Discrete",
          "Multiple",
          "data/152.csv",
          "readme/152.txt"
        ]
      ]
    }},
    {{
      "table_name": "SITES",
      "table_description": "Details of the sites where data is collected.",
      "column_names": [
        "SITE_ID",
        "SITE_NAME",
        "LOCATION",
        "CONTACT_EMAIL"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each site",
        "Name of the site",
        "Location of the site (e.g., West Branch, Iowa, United States)",
        "Contact email for inquiries about the site"
      ],
      "primary_key": [
        "SITE_ID"
      ],
      "sample_rows": [
        [
          1001,
          "West Branch, Iowa, United States (WBI)",
          "West Branch, Iowa, United States",
          "wbi@example.com"
        ],
        [
          1002,
          "Walnut Grove, California, United States (WGC)",
          "Walnut Grove, California, United States",
          "wgc@example.com"
        ]
      ]
    }},
    {{
      "table_name": "CATEGORIES",
      "table_description": "Categories of data collected (e.g., Greenhouse Gases, Air Quality).",
      "column_names": [
        "CATEGORY_ID",
        "CATEGORY_NAME",
        "DESCRIPTION"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each category",
        "Name of the category",
        "Description of the category"
      ],
      "primary_key": [
        "CATEGORY_ID"
      ],
      "sample_rows": [
        [
          1,
          "Greenhouse Gases",
          "Data related to greenhouse gases"
        ],
        [
          2,
          "Air Quality",
          "Data related to air quality"
        ]
      ]
    }},
    {{
      "table_name": "DATA_TYPES",
      "table_description": "Types of data collection methods used.",
      "column_names": [
        "TYPE_ID",
        "TYPE_NAME",
        "DESCRIPTION"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each data type",
        "Name of the data type (e.g., Surface PFP, Aircraft PFP)",
        "Description of the data type"
      ],
      "primary_key": [
        "TYPE_ID"
      ],
      "sample_rows": [
        [
          1,
          "Surface PFP",
          "Surface Profiler"
        ],
        [
          2,
          "Aircraft PFP",
          "Aircraft Profiler"
        ]
      ]
    }},
    {{
      "table_name": "FREQUENCIES",
      "table_description": "Frequencies of data collection.",
      "column_names": [
        "FREQUENCY_ID",
        "FREQUENCY_NAME",
        "DESCRIPTION"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each frequency",
        "Name of the frequency (e.g., Discrete, Continuous)",
        "Description of the frequency"
      ],
      "primary_key": [
        "FREQUENCY_ID"
      ],
      "sample_rows": [
        [
          1,
          "Discrete",
          "Data collected at specific intervals"
        ],
        [
          2,
          "Continuous",
          "Data collected continuously"
        ]
      ]
    }},
    {{
      "table_name": "YEARS",
      "table_description": "Years during which data was collected.",
      "column_names": [
        "YEAR_ID",
        "YEAR_NAME"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each year",
        "Year(s) the data was collected"
      ],
      "primary_key": [
        "YEAR_ID"
      ],
      "sample_rows": [
        [
          1,
          "Multiple"
        ],
        [
          2,
          "2023"
        ]
      ]
    }},
    {{
      "table_name": "DATA_FILES",
      "table_description": "Details of the data files associated with each dataset.",
      "column_names": [
        "FILE_ID",
        "DATASET_ID",
        "FILE_PATH",
        "FILE_SIZE",
        "UPLOAD_DATE"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "FLOAT",
        "DATE"
      ],
      "column_descriptions": [
        "Unique identifier for each data file",
        "ID of the dataset the file belongs to",
        "File path to the data file",
        "Size of the data file in MB",
        "Date the file was uploaded"
      ],
      "primary_key": [
        "FILE_ID"
      ],
      "sample_rows": [
        [
          1,
          151,
          "data/151.csv",
          1.2,
          "2023-01-01"
        ],
        [
          2,
          152,
          "data/152.csv",
          1.5,
          "2023-01-02"
        ]
      ]
    }},
    {{
      "table_name": "README_FILES",
      "table_description": "Details of the readme files associated with each dataset.",
      "column_names": [
        "README_ID",
        "DATASET_ID",
        "FILE_PATH",
        "FILE_SIZE",
        "UPLOAD_DATE"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "FLOAT",
        "DATE"
      ],
      "column_descriptions": [
        "Unique identifier for each readme file",
        "ID of the dataset the readme file belongs to",
        "File path to the readme file",
        "Size of the readme file in MB",
        "Date the file was uploaded"
      ],
      "primary_key": [
        "README_ID"
      ],
      "sample_rows": [
        [
          1,
          151,
          "readme/151.txt",
          0.1,
          "2023-01-01"
        ],
        [
          2,
          152,
          "readme/152.txt",
          0.1,
          "2023-01-02"
        ]
      ]
    }},
    {{
      "table_name": "USERS",
      "table_description": "Details of users accessing the datasets.",
      "column_names": [
        "USER_ID",
        "USER_NAME",
        "EMAIL",
        "ROLE"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each user",
        "Full name of the user",
        "Email address of the user",
        "Role of the user (e.g., researcher, data analyst, admin)"
      ],
      "primary_key": [
        "USER_ID"
      ],
      "sample_rows": [
        [
          1,
          "Alice Johnson",
          "alice.johnson@example.com",
          "researcher"
        ],
        [
          2,
          "Bob Williams",
          "bob.williams@example.com",
          "data analyst"
        ]
      ]
    }},
    {{
      "table_name": "ACCESS_LOGS",
      "table_description": "Tracks access to datasets by users.",
      "column_names": [
        "ACCESS_ID",
        "DATASET_ID",
        "USER_ID",
        "ACCESS_DATE",
        "ACCESS_TYPE"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "INTEGER",
        "DATE",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each access event",
        "ID of the dataset being accessed",
        "ID of the user accessing the dataset",
        "Date when the dataset was accessed",
        "Type of access (e.g., view, download)"
      ],
      "primary_key": [
        "ACCESS_ID"
      ],
      "sample_rows": [
        [
          1,
          151,
          1,
          "2023-05-01",
          "view"
        ],
        [
          2,
          152,
          2,
          "2023-05-02",
          "download"
        ]
      ]
    }}
  ],
  "foreign_keys": [
    {{
      "source_table": "DATASETS",
      "column_in_source_table": "SITE_ID",
      "referenced_table": "SITES",
      "column_in_referenced_table": "SITE_ID"
    }},
    {{
      "source_table": "DATASETS",
      "column_in_source_table": "CATEGORY",
      "referenced_table": "CATEGORIES",
      "column_in_referenced_table": "CATEGORY_ID"
    }},
    {{
      "source_table": "DATASETS",
      "column_in_source_table": "TYPE",
      "referenced_table": "DATA_TYPES",
      "column_in_referenced_table": "TYPE_ID"
    }},
    {{
      "source_table": "DATASETS",
      "column_in_source_table": "FREQUENCY",
      "referenced_table": "FREQUENCIES",
      "column_in_referenced_table": "FREQUENCY_ID"
    }},
    {{
      "source_table": "DATASETS",
      "column_in_source_table": "YEAR",
      "referenced_table": "YEARS",
      "column_in_referenced_table": "YEAR_ID"
    }},
    {{
      "source_table": "DATA_FILES",
      "column_in_source_table": "DATASET_ID",
      "referenced_table": "DATASETS",
      "column_in_referenced_table": "DATASET_ID"
    }},
    {{
      "source_table": "README_FILES",
      "column_in_source_table": "DATASET_ID",
      "referenced_table": "DATASETS",
      "column_in_referenced_table": "DATASET_ID"
    }},
    {{
      "source_table": "ACCESS_LOGS",
      "column_in_source_table": "DATASET_ID",
      "referenced_table": "DATASETS",
      "column_in_referenced_table": "DATASET_ID"
    }},
    {{
      "source_table": "ACCESS_LOGS",
      "column_in_source_table": "USER_ID",
      "referenced_table": "USERS",
      "column_in_referenced_table": "USER_ID"
    }}
  ]
}}        
        """)
    elif sample == 1:
        return ("All table and column names must be in lowercase.",
        """
{{
  "table_num": 5,
  "tables": [
    {{
      "table_name": "documents",
      "table_description": "Table to store details of each document uploaded to the platform.",
      "column_names": [
        "doc_id",
        "doc_title",
        "doc_version",
        "site_id",
        "upload_date",
        "last_updated",
        "hits",
        "author_id"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "INTEGER",
        "INTEGER",
        "DATE",
        "DATE",
        "INTEGER",
        "INTEGER"
      ],
      "column_descriptions": [
        "Unique identifier for each document",
        "Title of the document",
        "Version number of the document",
        "Reference to the site or department where the document is hosted",
        "Date the document was uploaded",
        "Date the document was last updated",
        "Number of views or hits the document has received",
        "ID of the author who uploaded the document"
      ],
      "primary_key": [
        "doc_id"
      ],
      "sample_rows": [
        [
          120983,
          "Canvas, How to Change the Course Name",
          1,
          101,
          "2022-08-01",
          "2022-08-31",
          821,
          3001
        ],
        [
          120334,
          "Contracts+ Using Jaggaer Word App",
          1,
          102,
          "203-04-01",
          "2023-04-24",
          822,
          3002
        ]
      ]
    }},
    {{
      "table_name": "sites",
      "table_description": "Details of the sites or departments hosting the documents.",
      "column_names": [
        "site_id",
        "site_name",
        "department",
        "contact_email"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each site",
        "Name of the site or department",
        "Department under which the site operates",
        "Contact email for inquiries about the site"
      ],
      "primary_key": [
        "site_id"
      ],
      "sample_rows": [
        [
          101,
          "University of Illinois Technology Services",
          "Technology Services",
          "techsupport@uillinois.edu"
        ],
        [
          102,
          "UI Training and Development Resources",
          "Training and Development",
          "train@uillinois.edu"
        ]
      ]
    }},
    {{
      "table_name": "authors",
      "table_description": "Information about the authors uploading the documents.",
      "column_names": [
        "authorid",
        "authorname",
        "email",
        "department"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each author",
        "Full name of the author",
        "Email address of the author",
        "Department the author belongs to"
      ],
      "primary_key": [
        "author_id"
      ],
      "sample_rows": [
        [
          3001,
          "John Doe",
          "jdoe@uillinois.edu",
          "Technology Services"
        ],
        [
          3002,
          "Jane Smith",
          "jsmith@uillinois.edu",
          "Training and Development"
        ]
      ]
    }},
    {{
      "table_name": "document_access",
      "table_description": "Tracks access to documents by users.",
      "column_names": [
        "access_id",
        "doc_id",
        "user_id",
        "access_date",
        "access_type"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "INTEGER",
        "DATE",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each access event",
        "ID of the document being accessed",
        "ID of the user accessing the document",
        "Date when the document was accessed",
        "Type of access (e.g., view, download)"
      ],
      "primary_key": [
        "access_id"
      ],
      "sample_rows": [
        [
          5001,
          120983,
          4001,
          "2023-05-01",
          "view"
        ],
        [
          5002,
          120334,
          4002,
          "2023-05-02",
          "download"
        ]
      ]
    }},
    {{
      "table_name": "users",
      "table_description": "Details of users accessing the documents.",
      "column_names": [
        "user_id",
        "user_name",
        "email",
        "role"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each user",
        "Full name of the user",
        "Email address of the user",
        "Role of the user (e.g., student, staff, admin)"
      ],
      "primary_key": [
        "user_id"
      ],
      "sample_rows": [
        [
          4001,
          "Alice Johnson",
          "alice.johnson@uillinois.edu",
          "student"
        ],
        [
          4002,
          "Bob Williams",
          "bob.williams@uillinois.edu",
          "staff"
        ]
      ]
    }}
  ],
  "foreign_keys": [
    {{
      "source_table": "documents",
      "column_in_source_table": "site_id",
      "referenced_table": "sites",
      "column_in_referenced_table": "site_id"
    }},
    {{
      "source_table": "documents",
      "column_in_source_table": "author_id",
      "referenced_table": "authors",
      "column_in_referenced_table": "authorid"
    }},
    {{
      "source_table": "document_access",
      "column_in_source_table": "doc_id",
      "referenced_table": "documents",
      "column_in_referenced_table": "doc_id"
    }},
    {{
      "source_table": "document_access",
      "column_in_source_table": "user_id",
      "referenced_table": "users",
      "column_in_referenced_table": "user_id"
    }}
  ]
}}        
        """,
        """
{{
  "table_num": 10,
  "tables": [
    {{
      "table_name": "datasets",
      "table_description": "Table to store details of each dataset collected from various sites.",
      "column_names": [
        "dataset_id",
        "site_id",
        "category",
        "name",
        "type",
        "frequency",
        "year",
        "data_file",
        "readme_file"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each dataset",
        "Reference to the site where the data was collected",
        "Category of the data (e.g., Greenhouse Gases)",
        "Name of the data (e.g., Carbon Dioxide(CO2))",
        "Type of data collection method (e.g., Surface PFP, Aircraft PFP)",
        "Frequency of data collection (e.g., Discrete, Continuous)",
        "Year(s) the data was collected",
        "File path to the data file",
        "File path to the readme file"
      ],
      "primary_key": [
        "dataset_id"
      ],
      "sample_rows": [
        [
          151,
          1001,
          "Greenhouse Gases",
          "Carbon Dioxide(CO2)",
          "Surface PFP",
          "Discrete",
          "Multiple",
          "data/151.csv",
          "readme/151.txt"
        ],
        [
          152,
          1002,
          "Greenhouse Gases",
          "Carbon Dioxide(CO2)",
          "Aircraft PFP",
          "Discrete",
          "Multiple",
          "data/152.csv",
          "readme/152.txt"
        ]
      ]
    }},
    {{
      "table_name": "sites",
      "table_description": "Details of the sites where data is collected.",
      "column_names": [
        "site_id",
        "site_name",
        "location",
        "contact_email"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each site",
        "Name of the site",
        "Location of the site (e.g., West Branch, Iowa, United States)",
        "Contact email for inquiries about the site"
      ],
      "primary_key": [
        "site_id"
      ],
      "sample_rows": [
        [
          1001,
          "West Branch, Iowa, United States (WBI)",
          "West Branch, Iowa, United States",
          "wbi@example.com"
        ],
        [
          1002,
          "Walnut Grove, California, United States (WGC)",
          "Walnut Grove, California, United States",
          "wgc@example.com"
        ]
      ]
    }},
    {{
      "table_name": "categories",
      "table_description": "Categories of data collected (e.g., Greenhouse Gases, Air Quality).",
      "column_names": [
        "category_id",
        "category_name",
        "description"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each category",
        "Name of the category",
        "Description of the category"
      ],
      "primary_key": [
        "category_id"
      ],
      "sample_rows": [
        [
          1,
          "Greenhouse Gases",
          "Data related to greenhouse gases"
        ],
        [
          2,
          "Air Quality",
          "Data related to air quality"
        ]
      ]
    }},
    {{
      "table_name": "data_types",
      "table_description": "Types of data collection methods used.",
      "column_names": [
        "type_id",
        "type_name",
        "description"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each data type",
        "Name of the data type (e.g., Surface PFP, Aircraft PFP)",
        "Description of the data type"
      ],
      "primary_key": [
        "type_id"
      ],
      "sample_rows": [
        [
          1,
          "Surface PFP",
          "Surface Profiler"
        ],
        [
          2,
          "Aircraft PFP",
          "Aircraft Profiler"
        ]
      ]
    }},
    {{
      "table_name": "frequencies",
      "table_description": "Frequencies of data collection.",
      "column_names": [
        "frequency_id",
        "frequency_name",
        "description"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each frequency",
        "Name of the frequency (e.g., Discrete, Continuous)",
        "Description of the frequency"
      ],
      "primary_key": [
        "frequency_id"
      ],
      "sample_rows": [
        [
          1,
          "Discrete",
          "Data collected at specific intervals"
        ],
        [
          2,
          "Continuous",
          "Data collected continuously"
        ]
      ]
    }},
    {{
      "table_name": "years",
      "table_description": "Years during which data was collected.",
      "column_names": [
        "year_id",
        "year_name"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each year",
        "Year(s) the data was collected"
      ],
      "primary_key": [
        "year_id"
      ],
      "sample_rows": [
        [
          1,
          "Multiple"
        ],
        [
          2,
          "2023"
        ]
      ]
    }},
    {{
      "table_name": "data_files",
      "table_description": "Details of the data files associated with each dataset.",
      "column_names": [
        "file_id",
        "dataset_id",
        "file_path",
        "file_size",
        "upload_date"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "FLOAT",
        "DATE"
      ],
      "column_descriptions": [
        "Unique identifier for each data file",
        "ID of the dataset the file belongs to",
        "File path to the data file",
        "Size of the data file in MB",
        "Date the file was uploaded"
      ],
      "primary_key": [
        "file_id"
      ],
      "sample_rows": [
        [
          1,
          151,
          "data/151.csv",
          1.2,
          "2023-01-01"
        ],
        [
          2,
          152,
          "data/152.csv",
          1.5,
          "2023-01-02"
        ]
      ]
    }},
    {{
      "table_name": "readme_files",
      "table_description": "Details of the readme files associated with each dataset.",
      "column_names": [
        "readme_id",
        "dataset_id",
        "file_path",
        "file_size",
        "upload_date"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "FLOAT",
        "DATE"
      ],
      "column_descriptions": [
        "Unique identifier for each readme file",
        "ID of the dataset the readme file belongs to",
        "File path to the readme file",
        "Size of the readme file in MB",
        "Date the file was uploaded"
      ],
      "primary_key": [
        "readme_id"
      ],
      "sample_rows": [
        [
          1,
          151,
          "readme/151.txt",
          0.1,
          "2023-01-01"
        ],
        [
          2,
          152,
          "readme/152.txt",
          0.1,
          "2023-01-02"
        ]
      ]
    }},
    {{
      "table_name": "users",
      "table_description": "Details of users accessing the datasets.",
      "column_names": [
        "user_id",
        "user_name",
        "email",
        "role"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each user",
        "Full name of the user",
        "Email address of the user",
        "Role of the user (e.g., researcher, data analyst, admin)"
      ],
      "primary_key": [
        "user_id"
      ],
      "sample_rows": [
        [
          1,
          "Alice Johnson",
          "alice.johnson@example.com",
          "researcher"
        ],
        [
          2,
          "Bob Williams",
          "bob.williams@example.com",
          "data analyst"
        ]
      ]
    }},
    {{
      "table_name": "access_logs",
      "table_description": "Tracks access to datasets by users.",
      "column_names": [
        "access_id",
        "dataset_id",
        "user_id",
        "access_date",
        "access_type"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "INTEGER",
        "DATE",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each access event",
        "ID of the dataset being accessed",
        "ID of the user accessing the dataset",
        "Date when the dataset was accessed",
        "Type of access (e.g., view, download)"
      ],
      "primary_key": [
        "access_id"
      ],
      "sample_rows": [
        [
          1,
          151,
          1,
          "2023-05-01",
          "view"
        ],
        [
          2,
          152,
          2,
          "2023-05-02",
          "download"
        ]
      ]
    }}
  ],
  "foreign_keys": [
    {{
      "source_table": "datasets",
      "column_in_source_table": "site_id",
      "referenced_table": "sites",
      "column_in_referenced_table": "site_id"
    }},
    {{
      "source_table": "datasets",
      "column_in_source_table": "category",
      "referenced_table": "categories",
      "column_in_referenced_table": "category_id"
    }},
    {{
      "source_table": "datasets",
      "column_in_source_table": "type",
      "referenced_table": "data_types",
      "column_in_referenced_table": "type_id"
    }},
    {{
      "source_table": "datasets",
      "column_in_source_table": "frequency",
      "referenced_table": "frequencies",
      "column_in_referenced_table": "frequency_id"
    }},
    {{
      "source_table": "datasets",
      "column_in_source_table": "year",
      "referenced_table": "years",
      "column_in_referenced_table": "year_id"
    }},
    {{
      "source_table": "data_files",
      "column_in_source_table": "dataset_id",
      "referenced_table": "datasets",
      "column_in_referenced_table": "dataset_id"
    }},
    {{
      "source_table": "readme_files",
      "column_in_source_table": "dataset_id",
      "referenced_table": "datasets",
      "column_in_referenced_table": "dataset_id"
    }},
    {{
      "source_table": "access_logs",
      "column_in_source_table": "dataset_id",
      "referenced_table": "datasets",
      "column_in_referenced_table": "dataset_id"
    }},
    {{
      "source_table": "access_logs",
      "column_in_source_table": "user_id",
      "referenced_table": "users",
      "column_in_referenced_table": "user_id"
    }}
  ]
}}        
        """
        )
    elif sample == 2:
        return ("All table and column names must have the first letter capitalized and the rest lowercase.",
        """
{{
  "table_num": 5,
  "tables": [
    {{
      "table_name": "Documents",
      "table_description": "Table to store details of each document uploaded to the platform.",
      "column_names": [
        "Doc_id",
        "Doc_title",
        "Doc_version",
        "Site_id",
        "Upload_date",
        "Last_updated",
        "Hits",
        "Author_id"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "INTEGER",
        "INTEGER",
        "DATE",
        "DATE",
        "INTEGER",
        "INTEGER"
      ],
      "column_descriptions": [
        "Unique identifier for each document",
        "Title of the document",
        "Version number of the document",
        "Reference to the site or department where the document is hosted",
        "Date the document was uploaded",
        "Date the document was last updated",
        "Number of views or hits the document has received",
        "ID of the author who uploaded the document"
      ],
      "primary_key": [
        "Doc_id"
      ],
      "sample_rows": [
        [
          120983,
          "Canvas, How to Change the Course Name",
          1,
          101,
          "2022-08-01",
          "2022-08-31",
          821,
          3001
        ],
        [
          120334,
          "Contracts+ Using Jaggaer Word App",
          1,
          102,
          "203-04-01",
          "2023-04-24",
          822,
          3002
        ]
      ]
    }},
    {{
      "table_name": "Sites",
      "table_description": "Details of the sites or departments hosting the documents.",
      "column_names": [
        "Site_id",
        "Site_name",
        "Department",
        "Contact_email"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each site",
        "Name of the site or department",
        "Department under which the site operates",
        "Contact email for inquiries about the site"
      ],
      "primary_key": [
        "Site_id"
      ],
      "sample_rows": [
        [
          101,
          "University of Illinois Technology Services",
          "Technology Services",
          "techsupport@uillinois.edu"
        ],
        [
          102,
          "UI Training and Development Resources",
          "Training and Development",
          "train@uillinois.edu"
        ]
      ]
    }},
    {{
      "table_name": "Authors",
      "table_description": "Information about the authors uploading the documents.",
      "column_names": [
        "Authorid",
        "Authorname",
        "Email",
        "Department"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each author",
        "Full name of the author",
        "Email address of the author",
        "Department the author belongs to"
      ],
      "primary_key": [
        "Author_id"
      ],
      "sample_rows": [
        [
          3001,
          "John Doe",
          "jdoe@uillinois.edu",
          "Technology Services"
        ],
        [
          3002,
          "Jane Smith",
          "jsmith@uillinois.edu",
          "Training and Development"
        ]
      ]
    }},
    {{
      "table_name": "Document_access",
      "table_description": "Tracks access to documents by users.",
      "column_names": [
        "Access_id",
        "Doc_id",
        "User_id",
        "Access_date",
        "Access_type"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "INTEGER",
        "DATE",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each access event",
        "ID of the document being accessed",
        "ID of the user accessing the document",
        "Date when the document was accessed",
        "Type of access (e.g., view, download)"
      ],
      "primary_key": [
        "Access_id"
      ],
      "sample_rows": [
        [
          5001,
          120983,
          4001,
          "2023-05-01",
          "view"
        ],
        [
          5002,
          120334,
          4002,
          "2023-05-02",
          "download"
        ]
      ]
    }},
    {{
      "table_name": "Users",
      "table_description": "Details of users accessing the documents.",
      "column_names": [
        "User_id",
        "User_name",
        "Email",
        "Role"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each user",
        "Full name of the user",
        "Email address of the user",
        "Role of the user (e.g., student, staff, admin)"
      ],
      "primary_key": [
        "User_id"
      ],
      "sample_rows": [
        [
          4001,
          "Alice Johnson",
          "alice.johnson@uillinois.edu",
          "student"
        ],
        [
          4002,
          "Bob Williams",
          "bob.williams@uillinois.edu",
          "staff"
        ]
      ]
    }}
  ],
  "foreign_keys": [
    {{
      "source_table": "Documents",
      "column_in_source_table": "Site_id",
      "referenced_table": "Sites",
      "column_in_referenced_table": "Site_id"
    }},
    {{
      "source_table": "Documents",
      "column_in_source_table": "Author_id",
      "referenced_table": "Authors",
      "column_in_referenced_table": "Authorid"
    }},
    {{
      "source_table": "Document_access",
      "column_in_source_table": "Doc_id",
      "referenced_table": "Documents",
      "column_in_referenced_table": "Doc_id"
    }},
    {{
      "source_table": "Document_access",
      "column_in_source_table": "User_id",
      "referenced_table": "Users",
      "column_in_referenced_table": "User_id"
    }}
  ]
}}
        """,
        """
{{
  "table_num": 10,
  "tables": [
    {{
      "table_name": "Datasets",
      "table_description": "Table to store details of each dataset collected from various sites.",
      "column_names": [
        "Dataset_id",
        "Site_id",
        "Category",
        "Name",
        "Type",
        "Frequency",
        "Year",
        "Data_file",
        "Readme_file"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each dataset",
        "Reference to the site where the data was collected",
        "Category of the data (e.g., Greenhouse Gases)",
        "Name of the data (e.g., Carbon Dioxide(CO2))",
        "Type of data collection method (e.g., Surface PFP, Aircraft PFP)",
        "Frequency of data collection (e.g., Discrete, Continuous)",
        "Year(s) the data was collected",
        "File path to the data file",
        "File path to the readme file"
      ],
      "primary_key": [
        "Dataset_id"
      ],
      "sample_rows": [
        [
          151,
          1001,
          "Greenhouse Gases",
          "Carbon Dioxide(CO2)",
          "Surface PFP",
          "Discrete",
          "Multiple",
          "data/151.csv",
          "readme/151.txt"
        ],
        [
          152,
          1002,
          "Greenhouse Gases",
          "Carbon Dioxide(CO2)",
          "Aircraft PFP",
          "Discrete",
          "Multiple",
          "data/152.csv",
          "readme/152.txt"
        ]
      ]
    }},
    {{
      "table_name": "Sites",
      "table_description": "Details of the sites where data is collected.",
      "column_names": [
        "Site_id",
        "Site_name",
        "Location",
        "Contact_email"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each site",
        "Name of the site",
        "Location of the site (e.g., West Branch, Iowa, United States)",
        "Contact email for inquiries about the site"
      ],
      "primary_key": [
        "Site_id"
      ],
      "sample_rows": [
        [
          1001,
          "West Branch, Iowa, United States (WBI)",
          "West Branch, Iowa, United States",
          "wbi@example.com"
        ],
        [
          1002,
          "Walnut Grove, California, United States (WGC)",
          "Walnut Grove, California, United States",
          "wgc@example.com"
        ]
      ]
    }},
    {{
      "table_name": "Categories",
      "table_description": "Categories of data collected (e.g., Greenhouse Gases, Air Quality).",
      "column_names": [
        "Category_id",
        "Category_name",
        "Description"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each category",
        "Name of the category",
        "Description of the category"
      ],
      "primary_key": [
        "Category_id"
      ],
      "sample_rows": [
        [
          1,
          "Greenhouse Gases",
          "Data related to greenhouse gases"
        ],
        [
          2,
          "Air Quality",
          "Data related to air quality"
        ]
      ]
    }},
    {{
      "table_name": "Data_types",
      "table_description": "Types of data collection methods used.",
      "column_names": [
        "Type_id",
        "Type_name",
        "Description"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each data type",
        "Name of the data type (e.g., Surface PFP, Aircraft PFP)",
        "Description of the data type"
      ],
      "primary_key": [
        "Type_id"
      ],
      "sample_rows": [
        [
          1,
          "Surface PFP",
          "Surface Profiler"
        ],
        [
          2,
          "Aircraft PFP",
          "Aircraft Profiler"
        ]
      ]
    }},
    {{
      "table_name": "Frequencies",
      "table_description": "Frequencies of data collection.",
      "column_names": [
        "Frequency_id",
        "Frequency_name",
        "Description"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each frequency",
        "Name of the frequency (e.g., Discrete, Continuous)",
        "Description of the frequency"
      ],
      "primary_key": [
        "Frequency_id"
      ],
      "sample_rows": [
        [
          1,
          "Discrete",
          "Data collected at specific intervals"
        ],
        [
          2,
          "Continuous",
          "Data collected continuously"
        ]
      ]
    }},
    {{
      "table_name": "Years",
      "table_description": "Years during which data was collected.",
      "column_names": [
        "Year_id",
        "Year_name"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each year",
        "Year(s) the data was collected"
      ],
      "primary_key": [
        "Year_id"
      ],
      "sample_rows": [
        [
          1,
          "Multiple"
        ],
        [
          2,
          "2023"
        ]
      ]
    }},
    {{
      "table_name": "Data_files",
      "table_description": "Details of the data files associated with each dataset.",
      "column_names": [
        "File_id",
        "Dataset_id",
        "File_path",
        "File_size",
        "Upload_date"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "FLOAT",
        "DATE"
      ],
      "column_descriptions": [
        "Unique identifier for each data file",
        "ID of the dataset the file belongs to",
        "File path to the data file",
        "Size of the data file in MB",
        "Date the file was uploaded"
      ],
      "primary_key": [
        "File_id"
      ],
      "sample_rows": [
        [
          1,
          151,
          "data/151.csv",
          1.2,
          "2023-01-01"
        ],
        [
          2,
          152,
          "data/152.csv",
          1.5,
          "2023-01-02"
        ]
      ]
    }},
    {{
      "table_name": "Readme_files",
      "table_description": "Details of the readme files associated with each dataset.",
      "column_names": [
        "Readme_id",
        "Dataset_id",
        "File_path",
        "File_size",
        "Upload_date"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "FLOAT",
        "DATE"
      ],
      "column_descriptions": [
        "Unique identifier for each readme file",
        "ID of the dataset the readme file belongs to",
        "File path to the readme file",
        "Size of the readme file in MB",
        "Date the file was uploaded"
      ],
      "primary_key": [
        "Readme_id"
      ],
      "sample_rows": [
        [
          1,
          151,
          "readme/151.txt",
          0.1,
          "2023-01-01"
        ],
        [
          2,
          152,
          "readme/152.txt",
          0.1,
          "2023-01-02"
        ]
      ]
    }},
    {{
      "table_name": "Users",
      "table_description": "Details of users accessing the datasets.",
      "column_names": [
        "User_id",
        "User_name",
        "Email",
        "Role"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each user",
        "Full name of the user",
        "Email address of the user",
        "Role of the user (e.g., researcher, data analyst, admin)"
      ],
      "primary_key": [
        "User_id"
      ],
      "sample_rows": [
        [
          1,
          "Alice Johnson",
          "alice.johnson@example.com",
          "researcher"
        ],
        [
          2,
          "Bob Williams",
          "bob.williams@example.com",
          "data analyst"
        ]
      ]
    }},
    {{
      "table_name": "Access_logs",
      "table_description": "Tracks access to datasets by users.",
      "column_names": [
        "Access_id",
        "Dataset_id",
        "User_id",
        "Access_date",
        "Access_type"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "INTEGER",
        "DATE",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each access event",
        "ID of the dataset being accessed",
        "ID of the user accessing the dataset",
        "Date when the dataset was accessed",
        "Type of access (e.g., view, download)"
      ],
      "primary_key": [
        "Access_id"
      ],
      "sample_rows": [
        [
          1,
          151,
          1,
          "2023-05-01",
          "view"
        ],
        [
          2,
          152,
          2,
          "2023-05-02",
          "download"
        ]
      ]
    }}
  ],
  "foreign_keys": [
    {{
      "source_table": "Datasets",
      "column_in_source_table": "Site_id",
      "referenced_table": "Sites",
      "column_in_referenced_table": "Site_id"
    }},
    {{
      "source_table": "Datasets",
      "column_in_source_table": "Category",
      "referenced_table": "Categories",
      "column_in_referenced_table": "Category_id"
    }},
    {{
      "source_table": "Datasets",
      "column_in_source_table": "Type",
      "referenced_table": "Data_types",
      "column_in_referenced_table": "Type_id"
    }},
    {{
      "source_table": "Datasets",
      "column_in_source_table": "Frequency",
      "referenced_table": "Frequencies",
      "column_in_referenced_table": "Frequency_id"
    }},
    {{
      "source_table": "Datasets",
      "column_in_source_table": "Year",
      "referenced_table": "Years",
      "column_in_referenced_table": "Year_id"
    }},
    {{
      "source_table": "Data_files",
      "column_in_source_table": "Dataset_id",
      "referenced_table": "Datasets",
      "column_in_referenced_table": "Dataset_id"
    }},
    {{
      "source_table": "Readme_files",
      "column_in_source_table": "Dataset_id",
      "referenced_table": "Datasets",
      "column_in_referenced_table": "Dataset_id"
    }},
    {{
      "source_table": "Access_logs",
      "column_in_source_table": "Dataset_id",
      "referenced_table": "Datasets",
      "column_in_referenced_table": "Dataset_id"
    }},
    {{
      "source_table": "Access_logs",
      "column_in_source_table": "User_id",
      "referenced_table": "Users",
      "column_in_referenced_table": "User_id"
    }}
  ]
}}        
        """
        )
    elif sample == 3:
        return ("Table names must be in uppercase, while column names must be in lowercase.",
        """
{{
  "table_num": 5,
  "tables": [
    {{
      "table_name": "DOCUMENTS",
      "table_description": "Table to store details of each document uploaded to the platform.",
      "column_names": [
        "doc_id",
        "doc_title",
        "doc_version",
        "site_id",
        "upload_date",
        "last_updated",
        "hits",
        "author_id"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "INTEGER",
        "INTEGER",
        "DATE",
        "DATE",
        "INTEGER",
        "INTEGER"
      ],
      "column_descriptions": [
        "Unique identifier for each document",
        "Title of the document",
        "Version number of the document",
        "Reference to the site or department where the document is hosted",
        "Date the document was uploaded",
        "Date the document was last updated",
        "Number of views or hits the document has received",
        "ID of the author who uploaded the document"
      ],
      "primary_key": [
        "doc_id"
      ],
      "sample_rows": [
        [
          120983,
          "Canvas, How to Change the Course Name",
          1,
          101,
          "2022-08-01",
          "2022-08-31",
          821,
          3001
        ],
        [
          120334,
          "Contracts+ Using Jaggaer Word App",
          1,
          102,
          "203-04-01",
          "2023-04-24",
          822,
          3002
        ]
      ]
    }},
    {{
      "table_name": "SITES",
      "table_description": "Details of the sites or departments hosting the documents.",
      "column_names": [
        "site_id",
        "site_name",
        "department",
        "contact_email"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each site",
        "Name of the site or department",
        "Department under which the site operates",
        "Contact email for inquiries about the site"
      ],
      "primary_key": [
        "site_id"
      ],
      "sample_rows": [
        [
          101,
          "University of Illinois Technology Services",
          "Technology Services",
          "techsupport@uillinois.edu"
        ],
        [
          102,
          "UI Training and Development Resources",
          "Training and Development",
          "train@uillinois.edu"
        ]
      ]
    }},
    {{
      "table_name": "AUTHORS",
      "table_description": "Information about the authors uploading the documents.",
      "column_names": [
        "authorid",
        "authorname",
        "email",
        "department"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each author",
        "Full name of the author",
        "Email address of the author",
        "Department the author belongs to"
      ],
      "primary_key": [
        "author_id"
      ],
      "sample_rows": [
        [
          3001,
          "John Doe",
          "jdoe@uillinois.edu",
          "Technology Services"
        ],
        [
          3002,
          "Jane Smith",
          "jsmith@uillinois.edu",
          "Training and Development"
        ]
      ]
    }},
    {{
      "table_name": "DOCUMENT_ACCESS",
      "table_description": "Tracks access to documents by users.",
      "column_names": [
        "access_id",
        "doc_id",
        "user_id",
        "access_date",
        "access_type"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "INTEGER",
        "DATE",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each access event",
        "ID of the document being accessed",
        "ID of the user accessing the document",
        "Date when the document was accessed",
        "Type of access (e.g., view, download)"
      ],
      "primary_key": [
        "access_id"
      ],
      "sample_rows": [
        [
          5001,
          120983,
          4001,
          "2023-05-01",
          "view"
        ],
        [
          5002,
          120334,
          4002,
          "2023-05-02",
          "download"
        ]
      ]
    }},
    {{
      "table_name": "USERS",
      "table_description": "Details of users accessing the documents.",
      "column_names": [
        "user_id",
        "user_name",
        "email",
        "role"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each user",
        "Full name of the user",
        "Email address of the user",
        "Role of the user (e.g., student, staff, admin)"
      ],
      "primary_key": [
        "user_id"
      ],
      "sample_rows": [
        [
          4001,
          "Alice Johnson",
          "alice.johnson@uillinois.edu",
          "student"
        ],
        [
          4002,
          "Bob Williams",
          "bob.williams@uillinois.edu",
          "staff"
        ]
      ]
    }}
  ],
  "foreign_keys": [
    {{
      "source_table": "DOCUMENTS",
      "column_in_source_table": "site_id",
      "referenced_table": "SITES",
      "column_in_referenced_table": "site_id"
    }},
    {{
      "source_table": "DOCUMENTS",
      "column_in_source_table": "author_id",
      "referenced_table": "AUTHORS",
      "column_in_referenced_table": "authorid"
    }},
    {{
      "source_table": "DOCUMENT_ACCESS",
      "column_in_source_table": "doc_id",
      "referenced_table": "DOCUMENTS",
      "column_in_referenced_table": "doc_id"
    }},
    {{
      "source_table": "DOCUMENT_ACCESS",
      "column_in_source_table": "user_id",
      "referenced_table": "USERS",
      "column_in_referenced_table": "user_id"
    }}
  ]
}}
        """,
        """
{{
  "table_num": 10,
  "tables": [
    {{
      "table_name": "DATASETS",
      "table_description": "Table to store details of each dataset collected from various sites.",
      "column_names": [
        "dataset_id",
        "site_id",
        "category",
        "name",
        "type",
        "frequency",
        "year",
        "data_file",
        "readme_file"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each dataset",
        "Reference to the site where the data was collected",
        "Category of the data (e.g., Greenhouse Gases)",
        "Name of the data (e.g., Carbon Dioxide(CO2))",
        "Type of data collection method (e.g., Surface PFP, Aircraft PFP)",
        "Frequency of data collection (e.g., Discrete, Continuous)",
        "Year(s) the data was collected",
        "File path to the data file",
        "File path to the readme file"
      ],
      "primary_key": [
        "dataset_id"
      ],
      "sample_rows": [
        [
          151,
          1001,
          "Greenhouse Gases",
          "Carbon Dioxide(CO2)",
          "Surface PFP",
          "Discrete",
          "Multiple",
          "data/151.csv",
          "readme/151.txt"
        ],
        [
          152,
          1002,
          "Greenhouse Gases",
          "Carbon Dioxide(CO2)",
          "Aircraft PFP",
          "Discrete",
          "Multiple",
          "data/152.csv",
          "readme/152.txt"
        ]
      ]
    }},
    {{
      "table_name": "SITES",
      "table_description": "Details of the sites where data is collected.",
      "column_names": [
        "site_id",
        "site_name",
        "location",
        "contact_email"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each site",
        "Name of the site",
        "Location of the site (e.g., West Branch, Iowa, United States)",
        "Contact email for inquiries about the site"
      ],
      "primary_key": [
        "site_id"
      ],
      "sample_rows": [
        [
          1001,
          "West Branch, Iowa, United States (WBI)",
          "West Branch, Iowa, United States",
          "wbi@example.com"
        ],
        [
          1002,
          "Walnut Grove, California, United States (WGC)",
          "Walnut Grove, California, United States",
          "wgc@example.com"
        ]
      ]
    }},
    {{
      "table_name": "CATEGORIES",
      "table_description": "Categories of data collected (e.g., Greenhouse Gases, Air Quality).",
      "column_names": [
        "category_id",
        "category_name",
        "description"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each category",
        "Name of the category",
        "Description of the category"
      ],
      "primary_key": [
        "category_id"
      ],
      "sample_rows": [
        [
          1,
          "Greenhouse Gases",
          "Data related to greenhouse gases"
        ],
        [
          2,
          "Air Quality",
          "Data related to air quality"
        ]
      ]
    }},
    {{
      "table_name": "DATA_TYPES",
      "table_description": "Types of data collection methods used.",
      "column_names": [
        "type_id",
        "type_name",
        "description"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each data type",
        "Name of the data type (e.g., Surface PFP, Aircraft PFP)",
        "Description of the data type"
      ],
      "primary_key": [
        "type_id"
      ],
      "sample_rows": [
        [
          1,
          "Surface PFP",
          "Surface Profiler"
        ],
        [
          2,
          "Aircraft PFP",
          "Aircraft Profiler"
        ]
      ]
    }},
    {{
      "table_name": "FREQUENCIES",
      "table_description": "Frequencies of data collection.",
      "column_names": [
        "frequency_id",
        "frequency_name",
        "description"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each frequency",
        "Name of the frequency (e.g., Discrete, Continuous)",
        "Description of the frequency"
      ],
      "primary_key": [
        "frequency_id"
      ],
      "sample_rows": [
        [
          1,
          "Discrete",
          "Data collected at specific intervals"
        ],
        [
          2,
          "Continuous",
          "Data collected continuously"
        ]
      ]
    }},
    {{
      "table_name": "YEARS",
      "table_description": "Years during which data was collected.",
      "column_names": [
        "year_id",
        "year_name"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each year",
        "Year(s) the data was collected"
      ],
      "primary_key": [
        "year_id"
      ],
      "sample_rows": [
        [
          1,
          "Multiple"
        ],
        [
          2,
          "2023"
        ]
      ]
    }},
    {{
      "table_name": "DATA_FILES",
      "table_description": "Details of the data files associated with each dataset.",
      "column_names": [
        "file_id",
        "dataset_id",
        "file_path",
        "file_size",
        "upload_date"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "FLOAT",
        "DATE"
      ],
      "column_descriptions": [
        "Unique identifier for each data file",
        "ID of the dataset the file belongs to",
        "File path to the data file",
        "Size of the data file in MB",
        "Date the file was uploaded"
      ],
      "primary_key": [
        "file_id"
      ],
      "sample_rows": [
        [
          1,
          151,
          "data/151.csv",
          1.2,
          "2023-01-01"
        ],
        [
          2,
          152,
          "data/152.csv",
          1.5,
          "2023-01-02"
        ]
      ]
    }},
    {{
      "table_name": "README_FILES",
      "table_description": "Details of the readme files associated with each dataset.",
      "column_names": [
        "readme_id",
        "dataset_id",
        "file_path",
        "file_size",
        "upload_date"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "FLOAT",
        "DATE"
      ],
      "column_descriptions": [
        "Unique identifier for each readme file",
        "ID of the dataset the readme file belongs to",
        "File path to the readme file",
        "Size of the readme file in MB",
        "Date the file was uploaded"
      ],
      "primary_key": [
        "readme_id"
      ],
      "sample_rows": [
        [
          1,
          151,
          "readme/151.txt",
          0.1,
          "2023-01-01"
        ],
        [
          2,
          152,
          "readme/152.txt",
          0.1,
          "2023-01-02"
        ]
      ]
    }},
    {{
      "table_name": "USERS",
      "table_description": "Details of users accessing the datasets.",
      "column_names": [
        "user_id",
        "user_name",
        "email",
        "role"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each user",
        "Full name of the user",
        "Email address of the user",
        "Role of the user (e.g., researcher, data analyst, admin)"
      ],
      "primary_key": [
        "user_id"
      ],
      "sample_rows": [
        [
          1,
          "Alice Johnson",
          "alice.johnson@example.com",
          "researcher"
        ],
        [
          2,
          "Bob Williams",
          "bob.williams@example.com",
          "data analyst"
        ]
      ]
    }},
    {{
      "table_name": "ACCESS_LOGS",
      "table_description": "Tracks access to datasets by users.",
      "column_names": [
        "access_id",
        "dataset_id",
        "user_id",
        "access_date",
        "access_type"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "INTEGER",
        "DATE",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each access event",
        "ID of the dataset being accessed",
        "ID of the user accessing the dataset",
        "Date when the dataset was accessed",
        "Type of access (e.g., view, download)"
      ],
      "primary_key": [
        "access_id"
      ],
      "sample_rows": [
        [
          1,
          151,
          1,
          "2023-05-01",
          "view"
        ],
        [
          2,
          152,
          2,
          "2023-05-02",
          "download"
        ]
      ]
    }}
  ],
  "foreign_keys": [
    {{
      "source_table": "DATASETS",
      "column_in_source_table": "site_id",
      "referenced_table": "SITES",
      "column_in_referenced_table": "site_id"
    }},
    {{
      "source_table": "DATASETS",
      "column_in_source_table": "category",
      "referenced_table": "CATEGORIES",
      "column_in_referenced_table": "category_id"
    }},
    {{
      "source_table": "DATASETS",
      "column_in_source_table": "type",
      "referenced_table": "DATA_TYPES",
      "column_in_referenced_table": "type_id"
    }},
    {{
      "source_table": "DATASETS",
      "column_in_source_table": "frequency",
      "referenced_table": "FREQUENCIES",
      "column_in_referenced_table": "frequency_id"
    }},
    {{
      "source_table": "DATASETS",
      "column_in_source_table": "year",
      "referenced_table": "YEARS",
      "column_in_referenced_table": "year_id"
    }},
    {{
      "source_table": "DATA_FILES",
      "column_in_source_table": "dataset_id",
      "referenced_table": "DATASETS",
      "column_in_referenced_table": "dataset_id"
    }},
    {{
      "source_table": "README_FILES",
      "column_in_source_table": "dataset_id",
      "referenced_table": "DATASETS",
      "column_in_referenced_table": "dataset_id"
    }},
    {{
      "source_table": "ACCESS_LOGS",
      "column_in_source_table": "dataset_id",
      "referenced_table": "DATASETS",
      "column_in_referenced_table": "dataset_id"
    }},
    {{
      "source_table": "ACCESS_LOGS",
      "column_in_source_table": "user_id",
      "referenced_table": "USERS",
      "column_in_referenced_table": "user_id"
    }}
  ]
}}        
        """
        )
    else:
        return ("All table and column names must follow the camel hump naming convention.",
        """
{{
  "table_num": 5,
  "tables": [
    {{
      "table_name": "documents",
      "table_description": "Table to store details of each document uploaded to the platform.",
      "column_names": [
        "docId",
        "docTitle",
        "docVersion",
        "siteId",
        "uploadDate",
        "lastUpdated",
        "hits",
        "authorId"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "INTEGER",
        "INTEGER",
        "DATE",
        "DATE",
        "INTEGER",
        "INTEGER"
      ],
      "column_descriptions": [
        "Unique identifier for each document",
        "Title of the document",
        "Version number of the document",
        "Reference to the site or department where the document is hosted",
        "Date the document was uploaded",
        "Date the document was last updated",
        "Number of views or hits the document has received",
        "ID of the author who uploaded the document"
      ],
      "primary_key": [
        "docId"
      ],
      "sample_rows": [
        [
          120983,
          "Canvas, How to Change the Course Name",
          1,
          101,
          "2022-08-01",
          "2022-08-31",
          821,
          3001
        ],
        [
          120334,
          "Contracts+ Using Jaggaer Word App",
          1,
          102,
          "203-04-01",
          "2023-04-24",
          822,
          3002
        ]
      ]
    }},
    {{
      "table_name": "sites",
      "table_description": "Details of the sites or departments hosting the documents.",
      "column_names": [
        "siteId",
        "siteName",
        "department",
        "contactEmail"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each site",
        "Name of the site or department",
        "Department under which the site operates",
        "Contact email for inquiries about the site"
      ],
      "primary_key": [
        "siteId"
      ],
      "sample_rows": [
        [
          101,
          "University of Illinois Technology Services",
          "Technology Services",
          "techsupport@uillinois.edu"
        ],
        [
          102,
          "UI Training and Development Resources",
          "Training and Development",
          "train@uillinois.edu"
        ]
      ]
    }},
    {{
      "table_name": "authors",
      "table_description": "Information about the authors uploading the documents.",
      "column_names": [
        "authorId",
        "authorName",
        "email",
        "department"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each author",
        "Full name of the author",
        "Email address of the author",
        "Department the author belongs to"
      ],
      "primary_key": [
        "authorId"
      ],
      "sample_rows": [
        [
          3001,
          "John Doe",
          "jdoe@uillinois.edu",
          "Technology Services"
        ],
        [
          3002,
          "Jane Smith",
          "jsmith@uillinois.edu",
          "Training and Development"
        ]
      ]
    }},
    {{
      "table_name": "documentAccess",
      "table_description": "Tracks access to documents by users.",
      "column_names": [
        "accessId",
        "docId",
        "userId",
        "accessDate",
        "accessType"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "INTEGER",
        "DATE",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each access event",
        "ID of the document being accessed",
        "ID of the user accessing the document",
        "Date when the document was accessed",
        "Type of access (e.g., view, download)"
      ],
      "primary_key": [
        "accessId"
      ],
      "sample_rows": [
        [
          5001,
          120983,
          4001,
          "2023-05-01",
          "view"
        ],
        [
          5002,
          120334,
          4002,
          "2023-05-02",
          "download"
        ]
      ]
    }},
    {{
      "table_name": "users",
      "table_description": "Details of users accessing the documents.",
      "column_names": [
        "userId",
        "userName",
        "email",
        "role"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each user",
        "Full name of the user",
        "Email address of the user",
        "Role of the user (e.g., student, staff, admin)"
      ],
      "primary_key": [
        "userId"
      ],
      "sample_rows": [
        [
          4001,
          "Alice Johnson",
          "alice.johnson@uillinois.edu",
          "student"
        ],
        [
          4002,
          "Bob Williams",
          "bob.williams@uillinois.edu",
          "staff"
        ]
      ]
    }}
  ],
  "foreign_keys": [
    {{
      "source_table": "documents",
      "column_in_source_table": "siteId",
      "referenced_table": "sites",
      "column_in_referenced_table": "siteId"
    }},
    {{
      "source_table": "documents",
      "column_in_source_table": "authorId",
      "referenced_table": "authors",
      "column_in_referenced_table": "authorId"
    }},
    {{
      "source_table": "documentAccess",
      "column_in_source_table": "docId",
      "referenced_table": "documents",
      "column_in_referenced_table": "docId"
    }},
    {{
      "source_table": "documentAccess",
      "column_in_source_table": "userId",
      "referenced_table": "users",
      "column_in_referenced_table": "userId"
    }}
  ]
}}
        """,
        """
{{
  "table_num": 10,
  "tables": [
    {{
      "table_name": "datasets",
      "table_description": "Table to store details of each dataset collected from various sites.",
      "column_names": [
        "datasetId",
        "siteId",
        "category",
        "name",
        "type",
        "frequency",
        "year",
        "dataFile",
        "readmeFile"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each dataset",
        "Reference to the site where the data was collected",
        "Category of the data (e.g., Greenhouse Gases)",
        "Name of the data (e.g., Carbon Dioxide(CO2))",
        "Type of data collection method (e.g., Surface PFP, Aircraft PFP)",
        "Frequency of data collection (e.g., Discrete, Continuous)",
        "Year(s) the data was collected",
        "File path to the data file",
        "File path to the readme file"
      ],
      "primary_key": [
        "datasetId"
      ],
      "sample_rows": [
        [
          151,
          1001,
          "Greenhouse Gases",
          "Carbon Dioxide(CO2)",
          "Surface PFP",
          "Discrete",
          "Multiple",
          "data/151.csv",
          "readme/151.txt"
        ],
        [
          152,
          1002,
          "Greenhouse Gases",
          "Carbon Dioxide(CO2)",
          "Aircraft PFP",
          "Discrete",
          "Multiple",
          "data/152.csv",
          "readme/152.txt"
        ]
      ]
    }},
    {{
      "table_name": "sites",
      "table_description": "Details of the sites where data is collected.",
      "column_names": [
        "siteId",
        "siteName",
        "location",
        "contactEmail"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each site",
        "Name of the site",
        "Location of the site (e.g., West Branch, Iowa, United States)",
        "Contact email for inquiries about the site"
      ],
      "primary_key": [
        "siteId"
      ],
      "sample_rows": [
        [
          1001,
          "West Branch, Iowa, United States (WBI)",
          "West Branch, Iowa, United States",
          "wbi@example.com"
        ],
        [
          1002,
          "Walnut Grove, California, United States (WGC)",
          "Walnut Grove, California, United States",
          "wgc@example.com"
        ]
      ]
    }},
    {{
      "table_name": "categories",
      "table_description": "Categories of data collected (e.g., Greenhouse Gases, Air Quality).",
      "column_names": [
        "categoryId",
        "categoryName",
        "description"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each category",
        "Name of the category",
        "Description of the category"
      ],
      "primary_key": [
        "categoryId"
      ],
      "sample_rows": [
        [
          1,
          "Greenhouse Gases",
          "Data related to greenhouse gases"
        ],
        [
          2,
          "Air Quality",
          "Data related to air quality"
        ]
      ]
    }},
    {{
      "table_name": "dataTypes",
      "table_description": "Types of data collection methods used.",
      "column_names": [
        "typeId",
        "typeName",
        "description"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each data type",
        "Name of the data type (e.g., Surface PFP, Aircraft PFP)",
        "Description of the data type"
      ],
      "primary_key": [
        "typeId"
      ],
      "sample_rows": [
        [
          1,
          "Surface PFP",
          "Surface Profiler"
        ],
        [
          2,
          "Aircraft PFP",
          "Aircraft Profiler"
        ]
      ]
    }},
    {{
      "table_name": "frequencies",
      "table_description": "Frequencies of data collection.",
      "column_names": [
        "frequencyId",
        "frequencyName",
        "description"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each frequency",
        "Name of the frequency (e.g., Discrete, Continuous)",
        "Description of the frequency"
      ],
      "primary_key": [
        "frequencyId"
      ],
      "sample_rows": [
        [
          1,
          "Discrete",
          "Data collected at specific intervals"
        ],
        [
          2,
          "Continuous",
          "Data collected continuously"
        ]
      ]
    }},
    {{
      "table_name": "years",
      "table_description": "Years during which data was collected.",
      "column_names": [
        "yearId",
        "yearName"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each year",
        "Year(s) the data was collected"
      ],
      "primary_key": [
        "yearId"
      ],
      "sample_rows": [
        [
          1,
          "Multiple"
        ],
        [
          2,
          "2023"
        ]
      ]
    }},
    {{
      "table_name": "dataFiles",
      "table_description": "Details of the data files associated with each dataset.",
      "column_names": [
        "fileId",
        "datasetId",
        "filePath",
        "fileSize",
        "uploadDate"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "FLOAT",
        "DATE"
      ],
      "column_descriptions": [
        "Unique identifier for each data file",
        "ID of the dataset the file belongs to",
        "File path to the data file",
        "Size of the data file in MB",
        "Date the file was uploaded"
      ],
      "primary_key": [
        "fileId"
      ],
      "sample_rows": [
        [
          1,
          151,
          "data/151.csv",
          1.2,
          "2023-01-01"
        ],
        [
          2,
          152,
          "data/152.csv",
          1.5,
          "2023-01-02"
        ]
      ]
    }},
    {{
      "table_name": "readmeFiles",
      "table_description": "Details of the readme files associated with each dataset.",
      "column_names": [
        "readmeId",
        "datasetId",
        "filePath",
        "fileSize",
        "uploadDate"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "VARCHAR",
        "FLOAT",
        "DATE"
      ],
      "column_descriptions": [
        "Unique identifier for each readme file",
        "ID of the dataset the readme file belongs to",
        "File path to the readme file",
        "Size of the readme file in MB",
        "Date the file was uploaded"
      ],
      "primary_key": [
        "readmeId"
      ],
      "sample_rows": [
        [
          1,
          151,
          "readme/151.txt",
          0.1,
          "2023-01-01"
        ],
        [
          2,
          152,
          "readme/152.txt",
          0.1,
          "2023-01-02"
        ]
      ]
    }},
    {{
      "table_name": "users",
      "table_description": "Details of users accessing the datasets.",
      "column_names": [
        "userId",
        "userName",
        "email",
        "role"
      ],
      "column_types": [
        "INTEGER",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each user",
        "Full name of the user",
        "Email address of the user",
        "Role of the user (e.g., researcher, data analyst, admin)"
      ],
      "primary_key": [
        "userId"
      ],
      "sample_rows": [
        [
          1,
          "Alice Johnson",
          "alice.johnson@example.com",
          "researcher"
        ],
        [
          2,
          "Bob Williams",
          "bob.williams@example.com",
          "data analyst"
        ]
      ]
    }},
    {{
      "table_name": "accessLogs",
      "table_description": "Tracks access to datasets by users.",
      "column_names": [
        "accessId",
        "datasetId",
        "userId",
        "accessDate",
        "accessType"
      ],
      "column_types": [
        "INTEGER",
        "INTEGER",
        "INTEGER",
        "DATE",
        "VARCHAR"
      ],
      "column_descriptions": [
        "Unique identifier for each access event",
        "ID of the dataset being accessed",
        "ID of the user accessing the dataset",
        "Date when the dataset was accessed",
        "Type of access (e.g., view, download)"
      ],
      "primary_key": [
        "accessId"
      ],
      "sample_rows": [
        [
          1,
          151,
          1,
          "2023-05-01",
          "view"
        ],
        [
          2,
          152,
          2,
          "2023-05-02",
          "download"
        ]
      ]
    }}
  ],
  "foreign_keys": [
    {{
      "source_table": "datasets",
      "column_in_source_table": "siteId",
      "referenced_table": "sites",
      "column_in_referenced_table": "siteId"
    }},
    {{
      "source_table": "datasets",
      "column_in_source_table": "category",
      "referenced_table": "categories",
      "column_in_referenced_table": "categoryId"
    }},
    {{
      "source_table": "datasets",
      "column_in_source_table": "type",
      "referenced_table": "dataTypes",
      "column_in_referenced_table": "typeId"
    }},
    {{
      "source_table": "datasets",
      "column_in_source_table": "frequency",
      "referenced_table": "frequencies",
      "column_in_referenced_table": "frequencyId"
    }},
    {{
      "source_table": "datasets",
      "column_in_source_table": "year",
      "referenced_table": "years",
      "column_in_referenced_table": "yearId"
    }},
    {{
      "source_table": "dataFiles",
      "column_in_source_table": "datasetId",
      "referenced_table": "datasets",
      "column_in_referenced_table": "datasetId"
    }},
    {{
      "source_table": "readmeFiles",
      "column_in_source_table": "datasetId",
      "referenced_table": "datasets",
      "column_in_referenced_table": "datasetId"
    }},
    {{
      "source_table": "accessLogs",
      "column_in_source_table": "datasetId",
      "referenced_table": "datasets",
      "column_in_referenced_table": "datasetId"
    }},
    {{
      "source_table": "accessLogs",
      "column_in_source_table": "userId",
      "referenced_table": "users",
      "column_in_referenced_table": "userId"
    }}
  ]
}}
        """
        )
    
if __name__ == '__main__':
    random.seed(42)
    tables = json.load(open("web_tables.json", "r", encoding = "utf-8"))
    prompt_template = open("./prompt_templates/schema_prompt.txt", "r", encoding = "utf-8").read()

    prompts = []
    for table in tables:
        random_table_num = generate_a_normal_integer()
        name_format, format_example_1, format_example_2 = generate_schema_form()
        print(random_table_num)
        prompt = prompt_template.format(
            table_num = random_table_num, 
            table = table,
            name_format = name_format,
            format_example_1 = format_example_1,
            format_example_2 = format_example_2
        )
        prompts.append(prompt.strip())

    random.shuffle(prompts)

    # Check whether output dir exists
    output_dir = "./prompts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"目录 {output_dir} 已创建")
    else:
        print(f"目录 {output_dir} 已存在")
    with open("./prompts/prompts_schema_synthesis.json", "w", encoding = "utf-8") as file:
        file.write(json.dumps(prompts, ensure_ascii = False, indent = 2))
