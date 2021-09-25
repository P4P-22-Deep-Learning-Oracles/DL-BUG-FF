package com.sirius.angular.service;


import com.sirius.angular.dto.Order;

import java.util.List;

public interface OrderService {
    List<Order> getOrders();
    Integer insert(Order order);
}
