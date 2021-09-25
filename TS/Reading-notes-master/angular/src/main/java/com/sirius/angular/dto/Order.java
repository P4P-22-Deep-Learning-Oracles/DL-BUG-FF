package com.sirius.angular.dto;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.sirius.angular.entity.Product;

import java.util.Date;
import java.util.List;

public class Order {
    private Integer id;
    private String name;
    private String street;
    private String city;
    private String state;
    private String zip;
    private String country;
    private Boolean giftwrap;
    @JsonFormat(pattern="yyyy-MM-dd HH:mm:ss", timezone = "GMT+08:00")
    private Date createTime;
    private List<Product> products;

    public void setOrder(com.sirius.angular.entity.Order order) {
        this.id = order.getId();
        this.name = order.getName();
        this.street = order.getStreet();
        this.city = order.getCity();
        this.state = order.getState();
        this.zip = order.getZip();
        this.country = order.getCountry();
        this.giftwrap = order.getGiftwrap();
        this.createTime = order.getCreateTime();
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getStreet() {
        return street;
    }

    public void setStreet(String street) {
        this.street = street;
    }

    public String getCity() {
        return city;
    }

    public void setCity(String city) {
        this.city = city;
    }

    public String getState() {
        return state;
    }

    public void setState(String state) {
        this.state = state;
    }

    public String getZip() {
        return zip;
    }

    public void setZip(String zip) {
        this.zip = zip;
    }

    public String getCountry() {
        return country;
    }

    public void setCountry(String country) {
        this.country = country;
    }

    public Boolean getGiftwrap() {
        return giftwrap;
    }

    public void setGiftwrap(Boolean giftwrap) {
        this.giftwrap = giftwrap;
    }

    public Date getCreateTime() {
        return createTime;
    }

    public void setCreateTime(Date createTime) {
        this.createTime = createTime;
    }

    public List<Product> getProducts() {
        return products;
    }

    public void setProducts(List<Product> products) {
        this.products = products;
    }
}
