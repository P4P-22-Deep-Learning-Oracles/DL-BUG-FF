import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import { NavbarComponent } from './navbar/navbar.component';
import { FooterComponent } from './footer/footer.component';
import { SearchComponent } from './search/search.component';
import { CarouseComponent } from './carouse/carouse.component';
import { ProductComponent } from './product/product.component';
import { StartsComponent } from './starts/starts.component';
import { ProductDetailComponent } from './product-detail/product-detail.component';
import { HomeComponent } from './home/home.component';
import {RouterModule, Routes} from "@angular/router";
import {ProductService} from "./shared/product.service";
import {FormsModule, ReactiveFormsModule} from "@angular/forms";
import { FilterPipe } from './pipe/filter.pipe';
import {HttpModule} from "@angular/http";


const routeConfig: Routes = [
  {path: '', component: HomeComponent},
  {path: 'product/:prodId', component: ProductDetailComponent}
];

@NgModule({
  declarations: [
    AppComponent,
    NavbarComponent,
    FooterComponent,
    SearchComponent,
    CarouseComponent,
    ProductComponent,
    StartsComponent,
    ProductDetailComponent,
    HomeComponent,
    FilterPipe
  ],
  imports: [
    BrowserModule,
    RouterModule.forRoot(routeConfig),
    ReactiveFormsModule,
    FormsModule,
    HttpModule
  ],
  providers: [ProductService],
  bootstrap: [AppComponent]
})
export class AppModule { }
