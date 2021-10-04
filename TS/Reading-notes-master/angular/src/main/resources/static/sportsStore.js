angular.module("sportsStore")
.constant("dataUrl", "http://47.90.107.224:8080/angular/Products")
.constant("orderUrl", "http://47.90.107.224:8080/angular/Order")
.controller("sportsStoreCtrl", function ($scope, $http, $location, dataUrl, orderUrl, cart) {
    
    $scope.data = {};

    $http.get(dataUrl)
    .success(function(data) {
        $scope.data.products = data;
    })
    .error(function(error){
        $scope.data.error = error;
        console.log(error);
    });

    $scope.sendOrder = function(shippingDetails) {
        var order = angular.copy(shippingDetails);
        order.products = cart.getProducts();

        $http.post(orderUrl, order)
        .success(function (data){
            $scope.data.orderId = data;
            cart.getProducts().length = 0;
        })
        .error(function (error) {
            $scope.data.orderError = error;
        })
        .finally(function(){
            $location.path("/complete");
        });
    }
        
});