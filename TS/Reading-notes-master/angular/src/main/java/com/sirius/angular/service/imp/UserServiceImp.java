package com.sirius.angular.service.imp;


import com.sirius.angular.entity.User;
import com.sirius.angular.mapper.UserMapper;
import com.sirius.angular.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service("com.sirius.angular.service.UserService")
public class UserServiceImp implements UserService {
    @Autowired
    private UserMapper userMapper;

    @Override
    public User getUserByUsername(String username) {
        return userMapper.selectByUsername(username);
    }
}
