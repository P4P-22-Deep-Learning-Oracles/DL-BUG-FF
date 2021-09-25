package com.sirius.angular.filter;

import com.sirius.angular.common.utils.JwtUtils;
import io.jsonwebtoken.Claims;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;

import javax.servlet.*;
import javax.servlet.annotation.WebFilter;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@WebFilter(filterName = "userFilter", urlPatterns = "/*")
public class UserFilter implements Filter {
    final Logger logger = LoggerFactory.getLogger(getClass());

    @Value("${jwt.base64Security}")
    private String base64Security;

    @Value("${jwt.ignore}")
    private Boolean ignore;

    public void init(FilterConfig filterConfig) throws ServletException {

    }

    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException,
            ServletException {
        HttpServletRequest httpRequest = (HttpServletRequest) request;
        HttpServletResponse httpResponse = (HttpServletResponse) response;
        String requestUri = httpRequest.getRequestURI();
        String contextPath = httpRequest.getContextPath();
        String relativeUri = requestUri.replace(contextPath, "");

        if (relativeUri.startsWith("/angular/")) {
            String auth = httpRequest.getHeader("Authorization");
            if ((auth != null) && (auth.length() > 7)) {
                String HeadStr = auth.substring(0, 6).toLowerCase();
                if (HeadStr.compareTo("bearer") == 0) {
                    auth = auth.substring(7, auth.length());
                    Claims claims = JwtUtils.parseJWT(auth, base64Security);
                    if (claims != null) {
                        logger.info(claims.toString());
                        httpRequest.setAttribute("roles", claims.get("roles"));
                        chain.doFilter(request, response);
                    } else {
                        httpRequest.getRequestDispatcher("/relogin.html").forward(httpRequest, httpResponse);
                    }
                }
            } else {
                if (ignore == true) {
                    chain.doFilter(request, response);
                } else {
                    httpRequest.getRequestDispatcher("/can_not_access.html").forward(httpRequest, httpResponse);
                }
            }
            return;
        }

        chain.doFilter(request, response);
    }

    public void destroy() {

    }
}
