PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE company_departments (
    department_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    budget REAL CHECK(budget >= 0),
    location TEXT,
    established_date DATE,
    description TEXT,
    is_active BOOLEAN DEFAULT 1,
    CONSTRAINT chk_location CHECK(location IN ('North', 'South', 'East', 'West', 'Central'))
);
INSERT INTO company_departments VALUES(1,'Engineering',1000000.0,'North','2010-01-15','Software development and engineering',1);
INSERT INTO company_departments VALUES(2,'Marketing',750000.0,'South','2011-03-22','Promotion and advertising',1);
INSERT INTO company_departments VALUES(3,'Finance',500000.0,'East','2010-01-10','Accounting and financial operations',1);
INSERT INTO company_departments VALUES(4,'HR',400000.0,'West','2012-05-18','Human resources management',1);
INSERT INTO company_departments VALUES(5,'IT Support',300000.0,'Central','2018-07-10','Technical support and infrastructure',1);
INSERT INTO company_departments VALUES(6,'Operations',850000.0,'North','2015-11-05','Business operations and logistics',1);
CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE CHECK(email LIKE '%@%.%'),
    phone TEXT CHECK(LENGTH(phone) BETWEEN 10 AND 15),
    hire_date DATE NOT NULL,
    job_title TEXT NOT NULL,
    salary REAL CHECK(salary > 0),
    commission_pct REAL CHECK(commission_pct BETWEEN 0 AND 1),
    manager_id INTEGER,
    department_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (department_id) REFERENCES company_departments(department_id) ON DELETE SET NULL,
    FOREIGN KEY (manager_id) REFERENCES employees(employee_id) ON DELETE SET NULL,
    CONSTRAINT full_name_unique UNIQUE (first_name, last_name)
);
INSERT INTO employees VALUES(1,'John','Smith','john.smith@company.com','1234567890','2015-06-10','CEO',250000.0,NULL,NULL,NULL,'2025-06-19 02:30:18','2025-06-19 02:30:18');
INSERT INTO employees VALUES(2,'Jane','Doe','jane.doe@company.com','2345678901','2016-02-15','CTO',220000.0,NULL,1,1,'2025-06-19 02:30:18','2025-06-19 02:30:18');
INSERT INTO employees VALUES(3,'Michael','Johnson','michael.johnson@company.com','3456789012','2017-03-20','Engineering Manager',180000.0,NULL,2,1,'2025-06-19 02:30:18','2025-06-19 02:30:18');
INSERT INTO employees VALUES(4,'Emily','Williams','emily.williams@company.com','4567890123','2018-04-25','Senior Developer',150000.0,NULL,3,1,'2025-06-19 02:30:18','2025-06-19 02:30:18');
INSERT INTO employees VALUES(5,'Robert','Brown','robert.brown@company.com','5678901234','2019-05-30','Marketing Director',160000.0,NULL,1,2,'2025-06-19 02:30:18','2025-06-19 02:30:18');
INSERT INTO employees VALUES(6,'Sarah','Miller','sarah.miller@company.com','6789012345','2020-06-05','Accountant',120000.0,NULL,1,3,'2025-06-19 02:30:18','2025-06-19 02:30:18');
INSERT INTO employees VALUES(7,'David','Wilson','david.wilson@company.com','7890123456','2021-07-10','HR Specialist',110000.0,NULL,1,4,'2025-06-19 02:30:18','2025-06-19 02:30:18');
INSERT INTO employees VALUES(8,'Lisa','Anderson','lisa.anderson@company.com','8901234567','2022-01-10','IT Support Specialist',85000.0,NULL,3,5,'2025-06-19 02:35:19','2025-06-19 02:35:19');
INSERT INTO employees VALUES(9,'Thomas','Clark','thomas.clark@company.com','9012345678','2021-09-15','Operations Manager',140000.0,NULL,1,6,'2025-06-19 02:35:19','2025-06-19 02:35:19');
INSERT INTO employees VALUES(10,'Jessica','White','jessica.white@company.com','0123456789','2023-02-20','Junior Developer',95000.0,NULL,3,1,'2025-06-19 02:35:19','2025-06-19 02:35:19');
INSERT INTO employees VALUES(11,'Daniel','Martin','daniel.martin@company.com','1234509876','2020-07-01','Marketing Specialist',105000.0,NULL,5,2,'2025-06-19 02:35:19','2025-06-19 02:35:19');
CREATE TABLE projects (
    project_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    start_date DATE,
    end_date DATE,
    budget REAL CHECK(budget >= 0),
    status TEXT CHECK(status IN ('Planning', 'In Progress', 'On Hold', 'Completed', 'Cancelled')),
    department_id INTEGER,
    FOREIGN KEY (department_id) REFERENCES company_departments(department_id) ON DELETE CASCADE,
    CONSTRAINT date_check CHECK (end_date IS NULL OR end_date >= start_date)
);
INSERT INTO projects VALUES(1,'Website Redesign','Complete overhaul of company website','2023-01-15','2023-06-30',50000.0,'In Progress',1);
INSERT INTO projects VALUES(2,'Product Launch','Launch of new product line','2023-02-01','2023-12-31',150000.0,'Planning',2);
INSERT INTO projects VALUES(3,'Financial System','Implementation of new accounting software','2023-03-01',NULL,75000.0,'In Progress',3);
INSERT INTO projects VALUES(4,'Employee Training','Quarterly employee training program','2023-04-01','2023-04-30',20000.0,'Completed',4);
INSERT INTO projects VALUES(5,'IT Infrastructure','Upgrade company IT infrastructure','2023-05-01','2023-12-31',120000.0,'In Progress',5);
INSERT INTO projects VALUES(6,'Supply Chain','Optimize supply chain operations','2023-06-15',NULL,95000.0,'Planning',6);
INSERT INTO projects VALUES(7,'Mobile App','Develop new company mobile app','2023-04-10','2023-09-30',80000.0,'In Progress',1);
CREATE TABLE employee_projects (
    employee_id INTEGER,
    project_id INTEGER,
    role TEXT NOT NULL,
    hours_worked REAL DEFAULT 0,
    start_date DATE NOT NULL,
    end_date DATE,
    PRIMARY KEY (employee_id, project_id),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);
INSERT INTO employee_projects VALUES(3,1,'Project Lead',120.5,'2023-01-15',NULL);
INSERT INTO employee_projects VALUES(4,1,'Developer',80.0,'2023-01-15',NULL);
INSERT INTO employee_projects VALUES(5,2,'Marketing Lead',60.0,'2023-02-01',NULL);
INSERT INTO employee_projects VALUES(6,3,'Finance Lead',45.5,'2023-03-01',NULL);
INSERT INTO employee_projects VALUES(7,4,'Training Coordinator',30.0,'2023-04-01',NULL);
INSERT INTO employee_projects VALUES(8,5,'Support Lead',45.0,'2023-05-01',NULL);
INSERT INTO employee_projects VALUES(9,6,'Project Manager',60.0,'2023-06-15',NULL);
INSERT INTO employee_projects VALUES(10,7,'Developer',75.5,'2023-04-10',NULL);
INSERT INTO employee_projects VALUES(11,5,'Technical Writer',30.0,'2023-05-15',NULL);
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE CHECK(email LIKE '%@%.%'),
    phone TEXT,
    address TEXT,
    city TEXT,
    state TEXT,
    zip_code TEXT,
    country TEXT DEFAULT 'USA',
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    credit_limit REAL CHECK(credit_limit >= 0),
    is_vip BOOLEAN DEFAULT 0
);
INSERT INTO customers VALUES(1,'Alice','Johnson','alice.johnson@example.com','5551234567','123 Main St','New York','NY','10001','USA','2025-06-19 02:30:18',5000.0,1);
INSERT INTO customers VALUES(2,'Bob','Smith','bob.smith@example.com','5552345678','456 Oak Ave','Los Angeles','CA','90001','USA','2025-06-19 02:30:18',2000.0,0);
INSERT INTO customers VALUES(3,'Charlie','Brown','charlie.brown@example.com','5553456789','789 Pine Rd','Chicago','IL','60601','USA','2025-06-19 02:30:18',1000.0,0);
INSERT INTO customers VALUES(4,'Diana','Miller','diana.miller@example.com','5554567890','321 Elm St','Houston','TX','77001','USA','2025-06-19 02:30:18',3000.0,1);
INSERT INTO customers VALUES(5,'Ethan','Garcia','ethan.garcia@example.com','5552223333','159 Market St','San Francisco','CA','94102','USA','2025-06-19 02:35:19',3500.0,0);
INSERT INTO customers VALUES(6,'Sophia','Martinez','sophia.martinez@example.com','5553334444','753 Commerce Dr','Seattle','WA','98101','USA','2025-06-19 02:35:19',4500.0,1);
INSERT INTO customers VALUES(7,'Mason','Robinson','mason.robinson@example.com','5554445555','852 Trade Ave','Boston','MA','02101','USA','2025-06-19 02:35:19',2000.0,0);
CREATE TABLE product_categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    parent_category_id INTEGER,
    FOREIGN KEY (parent_category_id) REFERENCES product_categories(category_id) ON DELETE SET NULL
);
INSERT INTO product_categories VALUES(1,'Electronics','Electronic devices and accessories',NULL);
INSERT INTO product_categories VALUES(2,'Clothing','Apparel and fashion items',NULL);
INSERT INTO product_categories VALUES(3,'Home & Garden','Home improvement and gardening',NULL);
INSERT INTO product_categories VALUES(4,'Laptops','Portable computers',1);
INSERT INTO product_categories VALUES(5,'Smartphones','Mobile phones',1);
INSERT INTO product_categories VALUES(6,'Men''s Clothing','Clothing for men',2);
INSERT INTO product_categories VALUES(7,'Women''s Clothing','Clothing for women',2);
INSERT INTO product_categories VALUES(8,'Tablets','Portable tablet devices',1);
INSERT INTO product_categories VALUES(9,'Accessories','Electronic accessories',1);
INSERT INTO product_categories VALUES(10,'Kids Clothing','Clothing for children',2);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    category_id INTEGER,
    price REAL CHECK(price > 0),
    cost REAL CHECK(cost >= 0),
    stock_quantity INTEGER CHECK(stock_quantity >= 0),
    reorder_level INTEGER DEFAULT 5,
    discontinued BOOLEAN DEFAULT 0,
    weight REAL CHECK(weight > 0),
    dimensions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES product_categories(category_id) ON DELETE SET NULL
);
INSERT INTO products VALUES(1,'UltraBook Pro','15.6" laptop with 16GB RAM',4,1299.990000000000009,800.0,50,5,0,3.5,'14 x 9 x 0.7 inches','2025-06-19 02:30:18','2025-06-19 02:30:18');
INSERT INTO products VALUES(2,'Galaxy Phone X','Latest smartphone with 128GB storage',5,899.990000000000009,550.0,100,5,0,0.4000000000000000222,'6.2 x 3 x 0.3 inches','2025-06-19 02:30:18','2025-06-19 02:30:18');
INSERT INTO products VALUES(3,'Men''s Casual Shirt','100% cotton button-down shirt',6,39.99000000000000198,15.0,200,5,0,0.5,NULL,'2025-06-19 02:30:18','2025-06-19 02:30:18');
INSERT INTO products VALUES(4,'Women''s Summer Dress','Lightweight floral dress',7,59.99000000000000198,25.0,150,5,0,0.5999999999999999778,NULL,'2025-06-19 02:30:18','2025-06-19 02:30:18');
INSERT INTO products VALUES(5,'Wireless Headphones','Noise-cancelling Bluetooth headphones',1,199.990000000000009,120.0,75,5,0,0.8000000000000000444,'7 x 7 x 3 inches','2025-06-19 02:30:18','2025-06-19 02:30:18');
INSERT INTO products VALUES(6,'Power Tablet','10.5" tablet with 64GB storage',8,499.990000000000009,300.0,40,5,0,1.199999999999999956,'9.5 x 6.8 x 0.3 inches','2025-06-19 02:35:19','2025-06-19 02:35:19');
INSERT INTO products VALUES(7,'Bluetooth Speaker','Portable wireless speaker',9,79.98999999999999489,40.0,120,5,0,0.9000000000000000222,'6 x 2 x 2 inches','2025-06-19 02:35:19','2025-06-19 02:35:19');
INSERT INTO products VALUES(8,'Kids T-Shirt','100% cotton t-shirt for kids',10,24.98999999999999844,8.0,180,5,0,0.2999999999999999889,NULL,'2025-06-19 02:35:19','2025-06-19 02:35:19');
INSERT INTO products VALUES(9,'Laptop Case','Protective case for 15.6" laptops',9,39.99000000000000198,15.0,90,5,0,0.5,'16 x 11 x 2 inches','2025-06-19 02:35:19','2025-06-19 02:35:19');
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    required_date TIMESTAMP,
    shipped_date TIMESTAMP,
    status TEXT CHECK(status IN ('Pending', 'Processing', 'Shipped', 'Delivered', 'Cancelled', 'Returned')),
    payment_method TEXT CHECK(payment_method IN ('Credit Card', 'PayPal', 'Bank Transfer', 'Cash on Delivery')),
    payment_status TEXT CHECK(payment_status IN ('Pending', 'Paid', 'Failed', 'Refunded')),
    shipping_fee REAL DEFAULT 0 CHECK(shipping_fee >= 0),
    tax_amount REAL DEFAULT 0 CHECK(tax_amount >= 0),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    CONSTRAINT shipped_date_check CHECK (shipped_date IS NULL OR shipped_date >= order_date),
    CONSTRAINT required_date_check CHECK (required_date IS NULL OR required_date >= order_date)
);
INSERT INTO orders VALUES(1,1,'2023-01-10 10:30:00','2023-01-15','2023-01-12','Delivered','Credit Card','Paid',9.99000000000000021,89.59999999999999431);
INSERT INTO orders VALUES(2,2,'2023-02-15 14:45:00','2023-02-20','2023-02-18','Shipped','PayPal','Paid',5.990000000000000213,23.98000000000000042);
INSERT INTO orders VALUES(3,3,'2023-03-20 09:15:00','2023-03-25',NULL,'Processing','Credit Card','Pending',0.0,19.98999999999999843);
INSERT INTO orders VALUES(4,1,'2023-04-05 16:20:00','2023-04-10',NULL,'Pending','Bank Transfer','Pending',9.99000000000000021,119.9500000000000028);
INSERT INTO orders VALUES(5,5,'2023-05-12 11:20:00','2023-05-17','2023-05-15','Delivered','PayPal','Paid',7.990000000000000213,31.19999999999999929);
INSERT INTO orders VALUES(6,6,'2023-06-08 14:30:00','2023-06-13',NULL,'Processing','Credit Card','Pending',5.990000000000000213,42.5);
INSERT INTO orders VALUES(7,7,'2023-06-15 09:45:00','2023-06-20',NULL,'Pending','Bank Transfer','Pending',0.0,23.98000000000000042);
CREATE TABLE order_details (
    order_detail_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER CHECK(quantity > 0),
    unit_price REAL CHECK(unit_price > 0),
    discount REAL DEFAULT 0 CHECK(discount BETWEEN 0 AND 1),
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE RESTRICT,
    CONSTRAINT unique_order_product UNIQUE (order_id, product_id)
);
INSERT INTO order_details VALUES(1,1,1,1,1299.990000000000009,0.0);
INSERT INTO order_details VALUES(2,1,5,1,199.990000000000009,0.1000000000000000055);
INSERT INTO order_details VALUES(3,2,3,2,39.99000000000000198,0.0);
INSERT INTO order_details VALUES(4,2,4,1,59.99000000000000198,0.1499999999999999945);
INSERT INTO order_details VALUES(5,3,2,1,899.990000000000009,0.0);
INSERT INTO order_details VALUES(6,4,5,3,199.990000000000009,0.2000000000000000111);
INSERT INTO order_details VALUES(7,5,8,1,499.990000000000009,0.05000000000000000277);
INSERT INTO order_details VALUES(8,5,10,2,24.98999999999999844,0.0);
INSERT INTO order_details VALUES(9,6,9,1,79.98999999999999489,0.0);
INSERT INTO order_details VALUES(10,6,11,1,39.99000000000000198,0.1000000000000000055);
INSERT INTO order_details VALUES(11,7,6,3,39.99000000000000198,0.1499999999999999945);
CREATE TABLE schools (
    school_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    address TEXT,
    phone TEXT,
    email TEXT CHECK(email LIKE '%@%.%'),
    principal_name TEXT,
    established_year INTEGER CHECK(established_year > 1800),
    accreditation TEXT,
    website TEXT
);
INSERT INTO schools VALUES(1,'Central High School','123 Education Ave, Springfield','555-123-4567','info@centralhigh.edu','Dr. James Wilson',1950,'State Accredited','www.centralhigh.edu');
INSERT INTO schools VALUES(2,'Riverside Academy','456 Learning Blvd, Rivertown','555-234-5678','contact@riverside.edu','Ms. Sarah Johnson',1975,'National Blue Ribbon','www.riverside.edu');
CREATE TABLE academic_years (
    academic_year_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    is_current BOOLEAN DEFAULT 0,
    school_id INTEGER NOT NULL,
    FOREIGN KEY (school_id) REFERENCES schools(school_id) ON DELETE CASCADE,
    CONSTRAINT date_order CHECK (end_date > start_date)
);
INSERT INTO academic_years VALUES(1,'one','2022-09-01','2023-06-15',1,1);
INSERT INTO academic_years VALUES(2,'two','2021-09-01','2022-06-15',0,1);
INSERT INTO academic_years VALUES(3,'three','2022-08-15','2023-05-30',1,2);
CREATE TABLE semesters (
    semester_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    academic_year_id INTEGER NOT NULL,
    FOREIGN KEY (academic_year_id) REFERENCES academic_years(academic_year_id) ON DELETE CASCADE,
    CONSTRAINT date_order CHECK (end_date > start_date),
    CONSTRAINT unique_semester_name UNIQUE (name, academic_year_id)
);
INSERT INTO semesters VALUES(1,'Fall 2022','2022-09-01','2022-12-20',1);
INSERT INTO semesters VALUES(2,'Spring 2023','2023-01-10','2023-06-15',1);
INSERT INTO semesters VALUES(3,'Fall 2022','2022-08-15','2022-12-15',3);
INSERT INTO semesters VALUES(4,'Spring 2023','2023-01-05','2023-05-30',3);
CREATE TABLE school_departments (
    department_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    school_id INTEGER NOT NULL,
    head_teacher_id INTEGER,
    budget REAL CHECK(budget >= 0),
    FOREIGN KEY (school_id) REFERENCES schools(school_id) ON DELETE CASCADE
);
INSERT INTO school_departments VALUES(1,'Mathematics',1,1,50000.0);
INSERT INTO school_departments VALUES(2,'Science1',1,2,60000.0);
INSERT INTO school_departments VALUES(3,'English',1,3,45000.0);
INSERT INTO school_departments VALUES(4,'Chinese',2,4,40000.0);
INSERT INTO school_departments VALUES(5,'Science2',2,5,55000.0);
INSERT INTO school_departments VALUES(6,'History',1,6,40000.0);
INSERT INTO school_departments VALUES(7,'Arts',2,7,35000.0);
CREATE TABLE teachers (
    teacher_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE CHECK(email LIKE '%@%.%'),
    phone TEXT,
    hire_date DATE NOT NULL,
    department_id INTEGER,
    salary REAL CHECK(salary > 0),
    qualification TEXT,
    specialization TEXT,
    is_active BOOLEAN DEFAULT 1,
    FOREIGN KEY (department_id) REFERENCES school_departments(department_id) ON DELETE SET NULL,
    CONSTRAINT full_name_unique UNIQUE (first_name, last_name)
);
INSERT INTO teachers VALUES(1,'Robert','Smith','r.smith@centralhigh.edu','555-111-2222','2010-08-15',1,65000.0,'PhD','Algebra',1);
INSERT INTO teachers VALUES(2,'Jennifer','Brown','j.brown@centralhigh.edu','555-222-3333','2015-09-01',2,70000.0,'MSc','Biology',1);
INSERT INTO teachers VALUES(3,'Michael','Davis','m.davis@centralhigh.edu','555-333-4444','2018-01-10',3,60000.0,'MA','Literature',1);
INSERT INTO teachers VALUES(4,'Sarah','Wilson','s.wilson@riverside.edu','555-444-5555','2012-08-20',4,62000.0,'PhD','Calculus',1);
INSERT INTO teachers VALUES(5,'David','Taylor','d.taylor@riverside.edu','555-555-6666','2019-09-05',5,68000.0,'MSc','Chemistry',1);
INSERT INTO teachers VALUES(6,'Elizabeth','Moore','e.moore@centralhigh.edu','555-666-7777','2017-08-25',6,58000.0,'MA','World History',1);
INSERT INTO teachers VALUES(7,'Christopher','Lee','c.lee@riverside.edu','555-777-8888','2020-09-10',7,55000.0,'MFA','Visual Arts',1);
INSERT INTO teachers VALUES(8,'Amanda','Harris','a.harris@centralhigh.edu','555-888-9999','2019-01-15',2,72000.0,'PhD','Physics',1);
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    date_of_birth DATE NOT NULL,
    gender TEXT CHECK(gender IN ('Male', 'Female', 'Other')),
    address TEXT,
    phone TEXT,
    email TEXT CHECK(email LIKE '%@%.%'),
    enrollment_date DATE NOT NULL,
    graduation_date DATE,
    is_active BOOLEAN DEFAULT 1,
    guardian_name TEXT,
    guardian_phone TEXT,
    CONSTRAINT age_check CHECK (date_of_birth < '2023-01-01'),
    CONSTRAINT graduation_check CHECK (graduation_date IS NULL OR graduation_date > enrollment_date)
);
INSERT INTO students VALUES(1,'Emily','Johnson','2007-05-12','Female','123 Student St, Springfield','555-666-7777','emily.j@student.edu','2021-09-01',NULL,1,'Mary Johnson','555-777-8888');
INSERT INTO students VALUES(2,'James','Williams','2006-08-23','Male','456 Scholar Ave, Springfield','555-777-9999','james.w@student.edu','2020-09-01',NULL,1,'Robert Williams','555-888-0000');
INSERT INTO students VALUES(3,'Olivia','Brown','2008-03-15','Female','789 Learner Rd, Rivertown','555-999-1111','olivia.b@student.edu','2022-08-15',NULL,1,'Lisa Brown','555-000-2222');
INSERT INTO students VALUES(4,'William','Jones','2007-11-30','Male','321 Study Ln, Rivertown','555-111-3333','william.j@student.edu','2021-08-15',NULL,1,'Thomas Jones','555-222-4444');
INSERT INTO students VALUES(5,'Ava','Garcia','2008-07-22','Female','456 Knowledge Ln, Springfield','555-333-4444','ava.g@student.edu','2022-09-01',NULL,1,'Carlos Garcia','555-444-5555');
INSERT INTO students VALUES(6,'Noah','Rodriguez','2007-04-18','Male','789 Wisdom Ave, Rivertown','555-444-6666','noah.r@student.edu','2021-08-15',NULL,1,'Maria Rodriguez','555-555-7777');
INSERT INTO students VALUES(7,'Isabella','Martinez','2009-01-30','Female','321 Insight St, Springfield','555-555-8888','isabella.m@student.edu','2023-09-01',NULL,1,'Juan Martinez','555-666-9999');
CREATE TABLE courses (
    course_id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    description TEXT,
    credits INTEGER CHECK(credits > 0),
    department_id INTEGER,
    prerequisite_course_id INTEGER,
    FOREIGN KEY (department_id) REFERENCES school_departments(department_id) ON DELETE SET NULL,
    FOREIGN KEY (prerequisite_course_id) REFERENCES courses(course_id) ON DELETE SET NULL
);
INSERT INTO courses VALUES(1,'MATH101','Algebra I','Introduction to algebraic concepts',4,1,NULL);
INSERT INTO courses VALUES(2,'MATH201','Algebra II','Advanced algebraic concepts',4,1,1);
INSERT INTO courses VALUES(3,'SCI101','Biology','Introduction to biological sciences',4,2,NULL);
INSERT INTO courses VALUES(4,'ENG101','English Literature','Survey of English literature',3,3,NULL);
INSERT INTO courses VALUES(5,'MATH102','Geometry','Fundamentals of geometry',4,4,NULL);
INSERT INTO courses VALUES(6,'SCI201','Chemistry','Principles of chemistry',4,5,NULL);
INSERT INTO courses VALUES(7,'HIST101','World History','Survey of world civilizations',3,6,NULL);
INSERT INTO courses VALUES(8,'ART101','Introduction to Art','Fundamentals of visual arts',3,7,NULL);
INSERT INTO courses VALUES(9,'SCI301','Physics','Principles of physics',4,2,NULL);
INSERT INTO courses VALUES(10,'MATH301','Calculus','Advanced calculus',4,1,NULL);
CREATE TABLE classes (
    class_id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id INTEGER NOT NULL,
    teacher_id INTEGER,
    semester_id INTEGER NOT NULL,
    room_number TEXT,
    schedule TEXT,
    max_capacity INTEGER CHECK(max_capacity > 0),
    current_enrollment INTEGER DEFAULT 0 CHECK(current_enrollment >= 0 AND current_enrollment <= max_capacity),
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE,
    FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE SET NULL,
    FOREIGN KEY (semester_id) REFERENCES semesters(semester_id) ON DELETE CASCADE
);
INSERT INTO classes VALUES(1,1,1,1,'A101','Mon/Wed 9:00-10:30',30,25);
INSERT INTO classes VALUES(2,3,2,1,'B205','Tue/Thu 10:00-11:30',25,20);
INSERT INTO classes VALUES(3,4,3,1,'C302','Mon/Wed/Fri 1:00-2:00',20,18);
INSERT INTO classes VALUES(4,5,4,3,'D104','Tue/Thu 8:30-10:00',25,22);
INSERT INTO classes VALUES(5,6,5,3,'E201','Mon/Wed/Fri 10:00-11:00',20,15);
INSERT INTO classes VALUES(6,7,6,2,'D201','Tue/Thu 1:00-2:30',25,18);
INSERT INTO classes VALUES(7,8,7,4,'E105','Mon/Wed 10:00-11:30',20,15);
INSERT INTO classes VALUES(8,9,8,2,'B305','Mon/Wed/Fri 9:00-10:00',30,22);
INSERT INTO classes VALUES(9,10,1,2,'A102','Tue/Thu 2:00-3:30',25,20);
CREATE TABLE enrollments (
    enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    enrollment_date DATE DEFAULT CURRENT_DATE,
    grade TEXT CHECK(grade IN ('A', 'B', 'C', 'D', 'F', 'I', 'W')),
    status TEXT CHECK(status IN ('Active', 'Dropped', 'Completed', 'Failed')),
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (class_id) REFERENCES classes(class_id) ON DELETE CASCADE,
    CONSTRAINT unique_enrollment UNIQUE (student_id, class_id)
);
INSERT INTO enrollments VALUES(1,1,1,'2022-09-01','B','Completed');
INSERT INTO enrollments VALUES(2,1,3,'2022-09-01','A','Completed');
INSERT INTO enrollments VALUES(3,2,1,'2022-09-01','A','Completed');
INSERT INTO enrollments VALUES(4,2,2,'2022-09-01','B','Completed');
INSERT INTO enrollments VALUES(5,3,4,'2022-08-15',NULL,'Active');
INSERT INTO enrollments VALUES(6,3,5,'2022-08-15',NULL,'Active');
INSERT INTO enrollments VALUES(7,4,4,'2022-08-15','C','Completed');
INSERT INTO enrollments VALUES(8,4,5,'2022-08-15',NULL,'Active');
INSERT INTO enrollments VALUES(9,5,6,'2023-01-10',NULL,'Active');
INSERT INTO enrollments VALUES(10,5,8,'2023-01-10',NULL,'Active');
INSERT INTO enrollments VALUES(11,6,7,'2023-01-05',NULL,'Active');
INSERT INTO enrollments VALUES(12,6,9,'2023-01-05','B','Completed');
INSERT INTO enrollments VALUES(13,7,6,'2023-09-01',NULL,'Active');
DELETE FROM sqlite_sequence;
INSERT INTO sqlite_sequence VALUES('company_departments',6);
INSERT INTO sqlite_sequence VALUES('employees',11);
INSERT INTO sqlite_sequence VALUES('projects',7);
INSERT INTO sqlite_sequence VALUES('customers',7);
INSERT INTO sqlite_sequence VALUES('product_categories',10);
INSERT INTO sqlite_sequence VALUES('products',9);
INSERT INTO sqlite_sequence VALUES('orders',7);
INSERT INTO sqlite_sequence VALUES('order_details',11);
INSERT INTO sqlite_sequence VALUES('schools',2);
INSERT INTO sqlite_sequence VALUES('academic_years',3);
INSERT INTO sqlite_sequence VALUES('semesters',4);
INSERT INTO sqlite_sequence VALUES('school_departments',7);
INSERT INTO sqlite_sequence VALUES('teachers',8);
INSERT INTO sqlite_sequence VALUES('students',7);
INSERT INTO sqlite_sequence VALUES('courses',10);
INSERT INTO sqlite_sequence VALUES('classes',9);
INSERT INTO sqlite_sequence VALUES('enrollments',13);
CREATE INDEX idx_employee_department ON employees(department_id);
CREATE INDEX idx_employee_manager ON employees(manager_id);
CREATE INDEX idx_project_department ON projects(department_id);
CREATE INDEX idx_employee_project ON employee_projects(employee_id, project_id);
CREATE INDEX idx_order_customer ON orders(customer_id);
CREATE INDEX idx_order_date ON orders(order_date);
CREATE INDEX idx_order_status ON orders(status);
CREATE INDEX idx_order_detail_order ON order_details(order_id);
CREATE INDEX idx_order_detail_product ON order_details(product_id);
CREATE INDEX idx_product_category ON products(category_id);
CREATE INDEX idx_student_active ON students(is_active);
CREATE INDEX idx_teacher_department ON teachers(department_id);
CREATE INDEX idx_class_course ON classes(course_id);
CREATE INDEX idx_class_teacher ON classes(teacher_id);
CREATE INDEX idx_class_semester ON classes(semester_id);
CREATE INDEX idx_enrollment_student ON enrollments(student_id);
CREATE INDEX idx_enrollment_class ON enrollments(class_id);
COMMIT;
