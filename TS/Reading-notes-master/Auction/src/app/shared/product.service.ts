import { Injectable } from '@angular/core';
import {Http} from "@angular/http";
import {Observable} from "rxjs/Observable";

@Injectable()
export class ProductService {

  private dataSource: Observable<any>;

  private products: Product[] = [
    new Product(1, '第1个商品名称', 1.99, 3.5, '这是第一个商品', ['电子商品', '电子产品']),
    new Product(2, '第2个商品名称', 2.99, 4.5, '这是第二个商品', ['电子商品', '电子产品']),
    new Product(3, '第3个商品名称', 2.99, 1.5, '这是第三个商品', ['电子商品', '电子产品']),
    new Product(4, '第4个商品名称', 12.99, 0.5, '这是第四个商品', ['电子商品', '电子产品']),
    new Product(5, '第5个商品名称', 22.99, 5, '这是第五个商品', ['电子商品', '电子产品']),
    new Product(6, '第6个商品名称', 50, 3.5, '这是第六个商品', ['电子商品', '电子产品'])
  ];

  private comments: Comment[] = [
    new Comment(1, 3, '2017-02-02 22:22:33', '张三', 1, '东西不错'),
    new Comment(2, 1, '2017-02-02 22:22:33', '张三', 4, '东西不错'),
    new Comment(3, 4, '2017-02-02 22:22:33', '张三', 3, '东西不错'),
    new Comment(4, 2, '2017-02-02 22:22:33', '张三', 5, '东西不错'),
    new Comment(5, 6, '2017-02-02 22:22:33', '张三', 2, '东西不错'),
    new Comment(6, 5, '2017-02-02 22:22:33', '张三', 2, '东西不错'),
    new Comment(7, 3, '2017-02-02 22:22:33', '张三', 1, '东西不错'),
    new Comment(8, 1, '2017-02-02 22:22:33', '张三', 4, '东西不错'),
    new Comment(9, 4, '2017-02-02 22:22:33', '张三', 3, '东西不错'),
    new Comment(10, 2, '2017-02-02 22:22:33', '张三', 5, '东西不错'),
    new Comment(11, 6, '2017-02-02 22:22:33', '张三', 2, '东西不错'),
    new Comment(12, 5, '2017-02-02 22:22:33', '张三', 2, '东西不错'),
    new Comment(13, 3, '2017-02-02 22:22:33', '张三', 1, '东西不错'),
    new Comment(14, 1, '2017-02-02 22:22:33', '张三', 4, '东西不错'),
    new Comment(15, 4, '2017-02-02 22:22:33', '张三', 3, '东西不错'),
    new Comment(16, 2, '2017-02-02 22:22:33', '张三', 5, '东西不错'),
    new Comment(17, 6, '2017-02-02 22:22:33', '张三', 2, '东西不错'),
    new Comment(18, 5, '2017-02-02 22:22:33', '张三', 2, '东西不错')
  ];

  constructor(private http:Http) {
    this.dataSource = this.http.get('/products').map((res) => res.json());
  }

  getProducts(): Product[] {
    return this.products;
  }

  getProduct(id: number): Product {
    this.dataSource.subscribe(
      (data) => this.products = data
    );

    return this.products.find((product: Product) => product.id == id);
  }

  getCommentForProductId(id: number): Comment[] {
    return this.comments.filter((comment: Comment) => comment.productId == id );
  }

  getAllCategories(): string[] {
    return ['电子产品', '日用品', '家电'];
  }

}

export class Product {
  constructor (
    public id: number,
    public title: string,
    public price: number,
    public rating: number,
    public desc: string,
    public categories: Array<string>
  ) {
  }
}

export class Comment {
  constructor(public id: number,
               public productId: number,
               public timestamp: string,
               public user: string,
               public rating: number,
               public content: string){

  }
}
