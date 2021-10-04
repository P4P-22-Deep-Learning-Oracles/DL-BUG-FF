package com.sirius.angular.common.utils;

import org.apache.commons.lang3.time.DateFormatUtils;
import org.apache.http.client.utils.DateUtils;

import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by bill on 2017-4-26.
 */
public class DateTimeUtils {
    public static final String DEFAULT_DATETIME_PATTERN = "yyyy-MM-dd HH:mm:ss";
    public static final String DEFAULT_DATE_PATTERN = "yyyy-MM-dd";

    public static String getCurrentDateTimeString() {
        SimpleDateFormat sdf =  new SimpleDateFormat(DEFAULT_DATETIME_PATTERN);
        return sdf.format(new Date());
    }

    public static String getCurrentDateString() {
        return DateFormatUtils.format(new Date(), DEFAULT_DATE_PATTERN);
    }

    public static String dateTimeToString(Date date) {
        SimpleDateFormat sdf =  new SimpleDateFormat(DEFAULT_DATETIME_PATTERN);
        return sdf.format(date);
    }

    public static Date stringToDateTime(String date) {
        return DateUtils.parseDate(date, new String[]{DEFAULT_DATETIME_PATTERN});
    }

    public static String dateToString(Date date) {
        SimpleDateFormat sdf =  new SimpleDateFormat(DEFAULT_DATETIME_PATTERN);
        return sdf.format(date);
    }

    public static Date stringToDate(String date) {
        return DateUtils.parseDate(date, new String[]{DEFAULT_DATE_PATTERN});
    }
}
