import {Component, OnInit} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Product, Comment, ProductService} from '../shared/product.service';

@Component({
  selector: 'app-product-detail',
  templateUrl: './product-detail.component.html',
  styleUrls: ['./product-detail.component.css']
})

export class ProductDetailComponent implements OnInit {

  private product: Product;
  private comments: Comment[];

  newRating = 5;
  newComment = '';
  isCommentHidden = true;

  constructor(private routeInfo: ActivatedRoute,
  private productService: ProductService) { }

  ngOnInit() {
    const prodId: number = this.routeInfo.snapshot.params['prodId'];
    this.product = this.productService.getProduct(prodId);
    this.comments = this.productService.getCommentForProductId(prodId);
  }

  addComment() {
    const comment = new Comment(0, this.product.id, new Date().toISOString(), 'someone', this.newRating, this.newComment);
    this.comments.unshift(comment);

    let sum: number;
    sum = this.comments.reduce((sum, comment) => sum + comment.rating, 0);
    this.product.rating = sum / this.comments.length;

    this.newRating = 5;
    this.newComment = '';
    this.isCommentHidden = true;
  }

}
