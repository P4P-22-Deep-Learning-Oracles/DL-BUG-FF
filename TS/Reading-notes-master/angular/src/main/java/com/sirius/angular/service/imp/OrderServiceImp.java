package com.sirius.angular.service.imp;

import com.sirius.angular.dto.Order;
import com.sirius.angular.entity.Product;
import com.sirius.angular.mapper.OrderMapper;
import com.sirius.angular.mapper.ProductMapper;
import com.sirius.angular.service.OrderService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Service("com.sirius.angular.service.OrderService")
public class OrderServiceImp implements OrderService{
    @Autowired
    private ProductMapper productMapper;

    @Autowired
    private OrderMapper orderMapper;

    @Override
    public List<Order> getOrders( ) {
        List<Order> orderList = new ArrayList<>();
        List<com.sirius.angular.entity.Order> orders = orderMapper.selectAll();

        for (com.sirius.angular.entity.Order o:orders) {
            Order order = new Order();
            order.setOrder(o);
            String products = o.getProducts();
            if (products != null && !products.isEmpty()) {
                String[] ids = products.split(",");
                List list = new ArrayList(Arrays.asList(ids));
                List<Product> productList = productMapper.selectByIds(list);
                order.setProducts(productList);
            }
            orderList.add(order);
        }
        return  orderList;
    }

    @Override
    public Integer insert(Order order) {
        Integer id = 0;
        com.sirius.angular.entity.Order record = new com.sirius.angular.entity.Order();
        record.setCity(order.getCity());
        record.setCountry(order.getCountry());
        record.setGiftwrap(order.getGiftwrap());
        record.setName(order.getName());
        record.setState(order.getState());
        record.setZip(order.getZip());
        record.setStreet(order.getStreet());

        List<Product> products = order.getProducts();
        String ids = "";
        for (int i = 0; i < products.size(); ++i) {
            ids += products.get(i).getId().toString();
            if (i == products.size()-1)break;
            ids += ",";
        }
        record.setProducts(ids);
        orderMapper.insertSelective(record);
        return record.getId();
    }
}
