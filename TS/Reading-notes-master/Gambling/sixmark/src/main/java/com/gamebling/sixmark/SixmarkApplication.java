package com.gamebling.sixmark;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.servlet.ServletComponentScan;

@SpringBootApplication
@ServletComponentScan
@MapperScan("com.gamebling.sixmark.mapper")
public class SixmarkApplication {

	public static void main(String[] args) {
		SpringApplication.run(SixmarkApplication.class, args);
	}
}
