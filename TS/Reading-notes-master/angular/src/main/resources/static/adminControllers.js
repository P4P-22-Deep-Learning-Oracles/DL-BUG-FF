angular.module("sportsStoreAdmin")
.constant("authUrl", "http://47.90.107.224:8080/users/login")
.constant("orderUrl", "http://47.90.107.224:8080/angular/Orders")
.controller("authCtrl", function($scope, $http, $location, authUrl)
{

    $scope.authenticate = function(user, pass) {
        $http.post(authUrl, {
            username: user,
            password: pass
        },{
            withCredentitals: true
        }).success(function(data){
            if (data.status == "SUCCESS") {
                $location.path("/main");
            } else {
                $scope.authenticationError=data;
            }
        }).error(function(error){
            $scope.authenticationError=error;
        });
    }
})
.controller("mainCtrl", function($scope){

    $scope.screens = ["Products", "Orders"];
    $scope.current = $scope.screens[0];

    $scope.setScreen = function(index) {
        $scope.current = $scope.screens[index];
    };

    $scope.getScreen = function() {
        return $scope.current == "Products" ? "/views/adminProducts.html" : "/views/adminOrders.html";
    };

})
.controller("ordersCtrl", function($scope, $http, orderUrl){
    $http.get(orderUrl, {withCredentitals: true})
    .success(function(data){
        if (data.status == "SUCCESS") {
            $scope.orders = data.data;
        } else {
            $scope.error = data;
        }
    })
    .error(function(error) {
        $scope.error = error;
    });

    $scope.selectedOrder;

    $scope.selectOrder = function(order) {
        console.log(JSON.stringify(order));
        $scope.selectedOrder = order;
    };

    $scope.calcTotal = function(order) {
        var total = 0;
        for (var i = 0; i < order.products.length; ++i) {
            total += order.products[i].price;
        }
        return total;
    };
});