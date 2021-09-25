angular.module("exampleApp", ["increment", "ngResource", "ngRoute", "ngAnimate"])
.constant("baseUrl", "http://47.90.107.224:8080/angular/Products/")
.factory("productsResource", function($resource, baseUrl) {
    return $resource(baseUrl + ":id", {id: "@id"}, {create: {method: "POST"}, save: {method: "PUT"}});
})
.config(function ($routeProvider, $locationProvider) {
    $locationProvider.html5Mode(true);

    $routeProvider.when("/list", {
        templateUrl: "/tableView.html"
    });

    $routeProvider.when("/edit/:id", {
        templateUrl: "/editView.html",
        controller: "editCtrl"
    });

    $routeProvider.when("/create/:id/:data*", {
        templateUrl: "/editView.html",
        controller: "editCtrl"
    });

    $routeProvider.otherwise({
        templateUrl: "tableView.html",
        controller: "tableCtrl",
        resolve: {
            data: function (productsResource) {
                return productsResource.query();
            }
        }
    });

})
.controller("defaultCtrl", function ($scope, $http, $resource, $location, $route, $routeParams, baseUrl) {
    $scope.displayMode = "list";
    $scope.currentProduct = null;

    // $scope.listProducts = function() {
    //     $scope.products = [
    //     {id: 0, name: "Dummy1", category: "Test", price: 1.25},
    //     {id: 1, name: "Dummy2", category: "Test", price: 2.45},
    //     {id: 2, name: "Dummy3", category: "Test", price: 4.25}
    //     ];
    // }

    $scope.productsResource = $resource(baseUrl + ":id", {id: "@id"});

    $scope.$on("$routeChangeSuccess", function(){
        if ($location,path().indexOf("/edit/") == 0) {
            var id = $routeParams["id"];
            for (var i = 0; i < $scope.products.length; ++i) {
                if ($scope.products[i].id == id) {
                    $scope.currentProduct = $scope.products[i];
                    break;
                }
            }
        }
    });


    $scope.listProducts = function () {

        // $http.get(baseUrl).then(function(response){
        //     $scope.products = response.data;
        // })
        // .catch(function(result){
        //     console.log("Http error: " + result);
        // });

        $scope.products = $scope.productsResource.query();


    }



    $scope.deleteProduct = function(product) {
        // $http({
        //     method: "DELETE",
        //     url: baseUrl + product.id
        // }).then(function () {
        //     $scope.products.splice($scope.products.indexOf(product), 1);
        // });

        product.$delete.then(function() {
            $scope.products.splice($scope.products.indexOf(product), 1);
        });
        // $scope.displayMode = "list";
        $location.path("/list");

    }

    $scope.createProduct = function(product) {
        // $http.post(baseUrl, product).then(function (newProduct) {
        //     $scope.products.push(product);
        //     $scope.displayMode = "list";
        // });

        new $scope.productsResource(product).$save().then(function(newProduct){
            $scope.products.push(newProduct);
            // $scope.displayMode = "list";
            $location.path("/list");
        });


    }

    $scope.updateProduct = function(product) {

        // $http({
        //     url: baseUrl + product.id,
        //     method: "PUT",
        //     data: product
        // }).then( function (modifiedProduct) {
        //     for (var i = 0; i < $scope.products.length; ++i) {
        //         if ($scope.products[i].id == modifiedProduct.id) {
        //             $scope.products[i] = modifiedProduct;
        //             break;
        //         }
        //     }
        //     $scope.displayMode = "list";
        // });

        product.$save();
        // $scope.displayMode = "list";
        $location.path("/list");
    }

    $scope.editOrCreateProduct = function (product) {
        // $scope.currentProduct = product ? angular.copy(product) : {};
        // $scope.displayMode = "edit";

        $scope.currentProduct = product ? product : {};
        // $scope.displayMode = "edit";
        $location.path("/edit");
    }

    $scope.saveEdit = function (product) {
        if (angular.isDefined(product.id)) {
            $scope.updateProduct(product);
        } else {
            $scope.createProduct(product);
        }
        $scope.currentProduct = {};
    } 

    $scope.cancelEdit = function() {

        if ($scope.currentProduct && $scope.currentProduct.$get) {
            $scope.currentProduct.$get();
        }

        $scope.currentProduct = {};
        // $scope.displayMode = "list";
        $location.path("/list");
    }

    $scope.listProducts();
})
.controller("tableCtrl", function($scope, $location, $route, data) {
    $scope.data.products = data;

    $scope.refreshProducts = function() {
        $route.reload();
    }
})
.controller("editCtrl", function($scope, $routeParams, $location){
    $scope.currentProduct = null;



    if ($location.path().indexOf("/edit/") == 0) {
        var id = $routeParams["id"];
        for (var i = 0; i < $scope.products.length; ++i) {
            if ($scope.products[i].id == id) {
                $scope.currentProduct = $scope.products[i];
                break;
            }
        }
    }

    $scope.cancelEdit = function () {
        // if ($scope.currentProduct && $scope.currentProduct.$get) {
        //     $scope.currentProduct.$get();
        // }
        // $scope.currentProduct = {};
        $location.path("/list");
    }

    $scope.updateProduct = function (product) {
        product.$save();
        $location.path("/list");
    }

    $scope.saveEdit = function (product) {
        if (angular.isDefined(product.id)) {
            $scope.updateProduct(product);
        } else {
            $scope.createProduct(product);
        }
        $scope.currentProduct = {};
    } 
});