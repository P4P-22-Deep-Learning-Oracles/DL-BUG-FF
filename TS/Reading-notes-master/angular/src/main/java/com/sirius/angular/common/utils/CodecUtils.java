package com.sirius.angular.common.utils;

import org.apache.commons.codec.digest.DigestUtils;

import java.util.UUID;

public class CodecUtils {
    public static String encrypt(String arg0) {
        return DigestUtils.sha1Hex(arg0);
    }

    public static String getUUID() {
        UUID uuid = UUID.randomUUID();
        String str = uuid.toString().toUpperCase();
        return str.replace("-", "");
    }
}
