import {Component, EventEmitter, Input, OnChanges, OnInit, Output, SimpleChanges} from '@angular/core';

@Component({
  selector: 'app-starts',
  templateUrl: './starts.component.html',
  styleUrls: ['./starts.component.css']
})
export class StartsComponent implements OnInit, OnChanges {

  @Output()
  private ratingChange: EventEmitter<number> = new EventEmitter();

  @Input()
  private rating = 0;

  @Input()
  private readonly: boolean = true;

  private stars: boolean[];

  constructor() {
  }

  ngOnInit() {
  }

  clickStar(index: number): void {
    if (! this.readonly) {
      this.rating = index + 1;
      this.ratingChange.emit(this.rating);
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    this.stars = [];
    for (let i = 1; i <= 5; ++i) {
      this.stars.push(i > this.rating);
    }
  }

}
