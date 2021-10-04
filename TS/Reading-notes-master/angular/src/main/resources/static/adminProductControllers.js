angular.module("sportsStoreAdmin")
.constant("productUrl", "http://47.90.107.224:8080/angular/Products/")
.config(function ($httpProvider) {
    $httpProvider.defaults.withCredentials = true;
})
.controller("productCtrl", function($scope, $resource, productUrl) {
    $scope.productsResource = $resource(productUrl + ":id", {id : "@id"});

    $scope.listProducts = function() {
        $scope.products = $scope.productsResource.query();
    }

    $scope.deleteProducts = function(product) {
        product.$delete().then(function(){
            $scope.products.splice($scope.products.indexOf(product), 1);
        });
    }

    $scope.createProduct = function (product) {
        new $scope.productsResource(product).$save().then(function(newProduct) {
            $scope.products.push(newProduct);
            $scope.editedProduct = null;
        });
    }

    $scope.updateProduct = function(product) {
        product.$save();
        $scope.editedProduct = null;
    }

    $scope.startEdit = function (product) {
        $scope.editedProduct = product;
    }

    $scope.cancelEdit = function () {
        $scope.editedProduct = null;
    }

    $scope.listProducts();
});
