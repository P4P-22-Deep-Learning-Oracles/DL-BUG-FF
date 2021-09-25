package com.sirius.angular.service;


import com.sirius.angular.entity.Product;

import java.util.List;

public interface ProductService {
    List<Product> getProducts();
    Integer addProduct(com.sirius.angular.dto.Product product);
}
