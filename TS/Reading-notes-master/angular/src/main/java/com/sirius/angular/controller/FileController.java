package com.sirius.angular.controller;

import com.sirius.angular.common.dto.ResponseDTO;
import com.sirius.angular.common.dto.ResponseMessage;
import com.sirius.angular.common.dto.ResponseStatus;
import com.sirius.angular.common.utils.CodecUtils;
import com.sirius.angular.common.utils.DateTimeUtils;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.net.ftp.FTPClient;
import org.apache.commons.net.ftp.FTPFile;
import org.apache.commons.net.ftp.FTPReply;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.MultipartHttpServletRequest;
import org.springframework.web.multipart.commons.CommonsMultipartResolver;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.Iterator;


@Controller
@RequestMapping("file")
public class FileController {
    Logger logger = LoggerFactory.getLogger(getClass());

    @Value("${ftp.hostname}")
    private String hostname;

    @Value("${ftp.port}")
    private int port;

    @Value("${ftp.username}")
    private String userName;

    @Value("${ftp.password}")
    private String password;

    @Value("${ftp.path}")
    private String path;


    @RequestMapping("/fileUpload")
    @ResponseBody
    public ResponseDTO<String> fileUpload(HttpServletRequest request, HttpServletResponse response) {
        ResponseDTO<String> responseDTO = new ResponseDTO<>();
        try {
            CommonsMultipartResolver multipartResolver = new CommonsMultipartResolver(request.getSession().getServletContext());
            if (multipartResolver.isMultipart(request)) {
                MultipartHttpServletRequest multiRequest = (MultipartHttpServletRequest) request;
                Iterator<String> fileNameList = multiRequest.getFileNames();
                while (fileNameList.hasNext()) {
                    long startTime = System.currentTimeMillis();
                    MultipartFile file = multiRequest.getFile(fileNameList.next());
                    if (file != null) {
                        String fileName = file.getOriginalFilename();
                        if (StringUtils.isNotEmpty(fileName)) {
                            logger.info("Upload file: {}", fileName);
                            String prefix = fileName.substring(fileName.lastIndexOf("."));
                            String uploadFileName = DateTimeUtils.getCurrentDateString() + "_" + CodecUtils.getUUID()
                                    + prefix;

                            String localFileFullName = request.getSession().getServletContext().getRealPath(uploadFileName);
                            File localFile = new File(localFileFullName);
                            file.transferTo(localFile);
                        }
                    }
                }
            }

        } catch (Exception e) {
            responseDTO.setStatus(ResponseStatus.FAIL);
            logger.error("File upload error", e);
        }
        return responseDTO;
    }

    @RequestMapping(value = "/fileDownload", method = RequestMethod.GET)
    public void fileDownload(HttpServletResponse response, @RequestParam("fileName") String fileName) {
        response.setHeader("Pragma", "No-cache");
        response.setHeader("Cache-Control", "no-cache");
        response.setDateHeader("Expires", 0);
        response.setContentType("image/jpeg");

        FTPClient ftpClient = new FTPClient();
        try {
            int reply;
            ftpClient.connect(hostname, port);
            ftpClient.login(userName, password);
            reply = ftpClient.getReplyCode();
            if (!FTPReply.isPositiveCompletion(reply)) {
                ftpClient.disconnect();
                return;
            }
            ftpClient.setFileType(ftpClient.BINARY_FILE_TYPE);
            ftpClient.changeWorkingDirectory(path);
            ftpClient.enterLocalPassiveMode();
            FTPFile[] fs = ftpClient.listFiles();
            for (int i = 0; i < fs.length; i++) {
                if (fs[i].getName().equals(fileName)) {
                    String saveAsFileName = new String(fs[i].getName().getBytes("UTF-8"), "ISO8859-1");
                    response.setHeader("Content-Disposition", "attachment;fileName=" + saveAsFileName);
                    OutputStream os = response.getOutputStream();
                    ftpClient.retrieveFile(fs[i].getName(), os);
                    os.flush();
                    os.close();
                    break;
                }
            }
            ftpClient.logout();
        } catch (Exception e) {
            logger.error("FTP Exception", e);
        } finally {
            if (ftpClient.isConnected()) {
                try {
                    ftpClient.disconnect();
                } catch (Exception e) {
                    logger.error("FTP Exception", e);
                }
            }
        }
    }
}
