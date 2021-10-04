package com.sirius.angular.controller;

import com.sirius.angular.common.dto.ResponseDTO;
import com.sirius.angular.common.dto.ResponseStatus;
import com.sirius.angular.dto.Order;
import com.sirius.angular.service.OrderService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

import javax.servlet.http.HttpServletRequest;
import java.util.List;

@Controller
@RequestMapping("angular")
public class OrderController {
    Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    @Qualifier("com.sirius.angular.service.OrderService")
    OrderService orderService;

    @RequestMapping(value = "/Orders", method = RequestMethod.GET)
    @ResponseBody
    public ResponseDTO<List<Order>> getOrders() {
        ResponseDTO<List<Order>> responseDTO = new ResponseDTO<>();

        try {
            List<Order> orders = orderService.getOrders();
            responseDTO.setData(orders);
            responseDTO.setStatus(ResponseStatus.SUCCESS);
            logger.info("get all orders success.");
        } catch (Exception e) {
            responseDTO.setStatus(ResponseStatus.FAIL);
            logger.info("get all orders failed.");
        }
        return  responseDTO;
    }

    @RequestMapping(value = "/Order", method = RequestMethod.POST, consumes = "application/json")
    @ResponseBody
    public ResponseDTO<Integer> postOrder(HttpServletRequest request, @RequestBody Order order) {
        ResponseDTO<Integer> responseDTO = new ResponseDTO<>();
        try {
            int id = orderService.insert(order);
            responseDTO.setData(id);
            responseDTO.setStatus(ResponseStatus.SUCCESS);
            logger.info("post all orders success.");
        } catch (Exception e) {
            responseDTO.setStatus(ResponseStatus.FAIL);
            logger.info("post all orders failed.");
        }
        return responseDTO;
    }
}
