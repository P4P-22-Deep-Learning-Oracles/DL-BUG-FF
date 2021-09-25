import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'filter'
})
export class FilterPipe implements PipeTransform {

  transform(list: any[], filterField: string, keywords: string): any[] {
    if (!filterField || !keywords) {
      return list;
    }
    return list.filter(
      item => { const filterValue = item[filterField];
      return filterValue.indexOf(keywords) >= 0; }
      );
  }
}
