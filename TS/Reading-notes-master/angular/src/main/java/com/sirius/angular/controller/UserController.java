package com.sirius.angular.controller;

import com.sirius.angular.common.dto.ResponseDTO;
import com.sirius.angular.common.dto.ResponseMessage;
import com.sirius.angular.common.dto.ResponseStatus;
import com.sirius.angular.dto.LoginUser;
import com.sirius.angular.entity.User;
import com.sirius.angular.service.UserService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

import javax.servlet.http.HttpServletRequest;

@Controller
@RequestMapping("users")
public class UserController {
    Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private UserService userService;

    @RequestMapping(value = "/login", method = RequestMethod.POST, consumes = "application/json")
    @ResponseBody
    public ResponseDTO<String> postOrder(HttpServletRequest request, @RequestBody LoginUser userDto) {
        ResponseDTO<String> responseDTO = new ResponseDTO<>();
        try {
            User user = userService.getUserByUsername(userDto.getUsername());
            if (user == null) {
                responseDTO.setStatus(ResponseStatus.FAIL);
                responseDTO.addMessage(new ResponseMessage("001", "your username is not exited."));
                logger.info("login username is not exited.");
            } else {
                if(!user.getPassword().equals(userDto.getPassword())) {
                    responseDTO.setStatus(ResponseStatus.FAIL);
                    responseDTO.addMessage(new ResponseMessage("002", "your password is wrong."));
                    logger.info("login password is wrong.");
                } else {
                    responseDTO.setStatus(ResponseStatus.SUCCESS);
                    logger.info("login success");
                }
            }
        } catch (Exception e) {
            responseDTO.setStatus(ResponseStatus.FAIL);
            logger.info("login failed.Exception has happened when select db.");
        }
        return responseDTO;
    }
}
