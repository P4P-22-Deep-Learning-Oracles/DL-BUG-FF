
SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";

DROP TABLE IF EXISTS `users`;
DROP TABLE IF EXISTS `product`;
DROP TABLE IF EXISTS `order`;

CREATE TABLE IF NOT EXISTS `users` (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT 'ID',
  `username` varchar(64) DEFAULT NULL COMMENT '用户名称',
  `password` varchar(64) DEFAULT NULL COMMENT '密码',
  `create_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP COMMENT '生成时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='用户列表信息' AUTO_INCREMENT=1;

INSERT INTO `users` (`username`, `password`, `create_time`) VALUES
( 'admin', '123456', CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS `product` (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT 'id',
  `name` varchar(64) NOT NULL COMMENT '名字',
  `description` varchar(64) NOT NULL COMMENT '描述',
  `category` varchar(64) NOT NULL COMMENT '类型',
  `price` decimal(10,2) NOT NULL COMMENT '价格',
  `create_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP COMMENT '生成时间',
  PRIMARY KEY (`id`),
  UNIQUE(`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='保险类型信息' AUTO_INCREMENT=1;

INSERT INTO `product`(`name`, `description`, `category`, `price`, `create_time`) VALUES 
('Kayak', 'A boat for one person', 'Watersports', 275, CURRENT_TIMESTAMP),
('Lifejacket', 'Protective and fashionable', 'Watersports', 48.95, CURRENT_TIMESTAMP),
('Soccer Ball', 'FIFA-approved size and weigth', 'Soccer', 19.5, CURRENT_TIMESTAMP),
('Corner Flags', 'Give your playing field a professional touch', 'Soccer', 34.95, CURRENT_TIMESTAMP),
('Stadium', 'Flat-packed 35,000-seat stadium', 'Soccer', 79500.00, CURRENT_TIMESTAMP),
('Thinking Cap', 'Improve your brain efficiency by 75%', 'chess', 16, CURRENT_TIMESTAMP),
('Unsteady Chair', 'Secretly give your opponent a disadvantage', 'chess', 29.95, CURRENT_TIMESTAMP),
('Humam Chess Board', 'A fun game for the family', 'chess', 75, CURRENT_TIMESTAMP),
('Bliing-Bling King', 'Gold-plated, diamond-studded King', 'chess', 75, CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS `order` (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT 'ID',
  `name` varchar(64) DEFAULT NULL COMMENT '名字',
  `street` varchar(64) DEFAULT NULL COMMENT '街道',
  `city` varchar(64) DEFAULT NULL COMMENT '城市',
  `state` varchar(64) DEFAULT NULL COMMENT '州',
  `zip` varchar(64) DEFAULT NULL COMMENT '邮编',
  `country` varchar(64) DEFAULT NULL COMMENT '国家',
  `giftwrap` tinyint(1) DEFAULT '0' COMMENT '礼品包装',
  `products` varchar(128) DEFAULT NULL COMMENT '产品列表',
  `create_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP COMMENT '生成时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='订单信息' AUTO_INCREMENT=1;


INSERT INTO `order` (`name`, `street`, `city`, `state`, `zip`, `country`, `giftwrap`, `products`, `create_time`) VALUES
('zhazha3', 'street', 'city', 'state', 'zip', 'country', 1, '1', CURRENT_TIMESTAMP);
