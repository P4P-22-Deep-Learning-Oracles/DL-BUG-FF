package com.sirius.angular.service;


import com.sirius.angular.entity.User;

public interface UserService {
    User getUserByUsername(String username);
}
