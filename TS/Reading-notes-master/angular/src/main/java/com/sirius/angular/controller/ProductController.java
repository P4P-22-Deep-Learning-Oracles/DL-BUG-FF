package com.sirius.angular.controller;

import com.sirius.angular.common.dto.ResponseDTO;
import com.sirius.angular.common.dto.ResponseStatus;
import com.sirius.angular.entity.Product;
import com.sirius.angular.service.ProductService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.List;

@Controller
@RequestMapping("angular")
public class ProductController {
    Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    @Qualifier("com.sirius.angular.service.ProductService")
    ProductService productService;

    @RequestMapping(value = "/Products", method = RequestMethod.GET)
    @ResponseBody
    public List<Product> getProducts() {
        ResponseDTO<List<Product>> responseDTO = new ResponseDTO<>();

        try {
            List<Product> products = productService.getProducts();
            responseDTO.setData(products);
            responseDTO.setStatus(ResponseStatus.SUCCESS);
            logger.info("get all Products success.");
        } catch (Exception e) {
            responseDTO.setStatus(ResponseStatus.FAIL);
            logger.info("get all Products failed.");
        }

        return  responseDTO.getData();
    }

    @RequestMapping(value = "/Products", method = RequestMethod.POST, consumes = "application/json")
    @ResponseBody
    public Integer addProducts(@RequestBody com.sirius.angular.dto.Product product) {
        Integer id = -1;
        try {
            id = productService.addProduct(product);
            logger.info("add a new Product success.");
        } catch (Exception e) {
            logger.info("add a new Product failed.");
        }
        return  id;
    }
}
