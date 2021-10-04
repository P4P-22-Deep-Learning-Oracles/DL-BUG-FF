package com.sirius.angular.common.dto;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.ArrayList;
import java.util.List;

@JsonInclude(JsonInclude.Include.NON_NULL)
public class ResponseDTO<T> {
    private T data;
    private ResponseStatus status;
    private List<ResponseMessage> messageList;

    public T getData() {
        return data;
    }

    public void setData(T data) {
        this.data = data;
    }

    public ResponseStatus getStatus() {
        return status;
    }

    public void setStatus(ResponseStatus status) {
        this.status = status;
    }

    public List<ResponseMessage> getMessageList() {
        return messageList;
    }

    public void setMessageList(List<ResponseMessage> messageList) {
        this.messageList = messageList;
    }

    public void addMessage(ResponseMessage ret) {
        if (this.messageList == null) {
            this.messageList = new ArrayList();
        }
        this.messageList.add(ret);
    }
}
