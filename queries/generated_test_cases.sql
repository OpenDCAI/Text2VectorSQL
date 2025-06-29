-- 基于 data/test_vec.sql 中创建的表的 sqlite-vec 测试用例

-- === 产品表 (products) 测试 ===

-- 测试 1: 查找与“无线蓝牙耳机”最相似的5个产品
-- 使用 rowid=1 的产品 '无线蓝牙耳机' 的向量进行查询
SELECT rowid, name, price, category, distance
FROM products
WHERE product_embedding MATCH '[0.12,0.45,-0.23,0.67,-0.89,0.34,-0.56,0.78]'
ORDER BY distance
LIMIT 5;

-- 测试 2: 查找与“无线蓝牙耳机”最相似的3个电子产品 (electronics)
-- 结合了向量搜索和标量过滤 (WHERE category = 'electronics')
SELECT rowid, name, price, category, distance
FROM products
WHERE product_embedding MATCH '[0.12,0.45,-0.23,0.67,-0.89,0.34,-0.56,0.78]' AND category = 'electronics'
ORDER BY distance
LIMIT 3;


-- === 用户表 (users) 测试 ===

-- 测试 3: 查找与用户 'AliceSmith' 最相似的5个用户
-- 使用 rowid=1 的用户 'AliceSmith' 的向量进行查询
SELECT rowid, username, age, gender, distance
FROM users
WHERE user_embedding MATCH '[0.10,-0.20,0.30,-0.40,0.50,-0.60,0.70,-0.80,0.90,-0.10,0.20,-0.30,0.40,-0.50,0.60,-0.70]'
ORDER BY distance
LIMIT 5;

-- 测试 4: 查找与用户 'AliceSmith' 最相似的5个女性用户
-- 结合了向量搜索和标量过滤 (WHERE gender = 'female')
SELECT rowid, username, age, gender, distance
FROM users
WHERE user_embedding MATCH '[0.10,-0.20,0.30,-0.40,0.50,-0.60,0.70,-0.80,0.90,-0.10,0.20,-0.30,0.40,-0.50,0.60,-0.70]' AND gender = 'female'
ORDER BY distance
LIMIT 5;


-- === 地理位置表 (locations) 测试 ===

-- 测试 5: 查找离“天安门广场”最近的5个地点
-- 使用 rowid=1 的地点 '天安门广场' 的坐标进行查询
SELECT rowid, place_name, city, country, distance
FROM locations
WHERE coordinates MATCH '[116.404,39.915,50.0,0.01]'
ORDER BY distance
LIMIT 5;

-- 测试 6: 查找离“天安门广场”最近的5个地点，但不包括中国的地点
-- 结合了向量搜索和标量过滤 (WHERE country != '中国')
SELECT rowid, place_name, city, country, distance
FROM locations
WHERE coordinates MATCH '[116.404,39.915,50.0,0.01]' AND country != '中国'
ORDER BY distance
LIMIT 5;
