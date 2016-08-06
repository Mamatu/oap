/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.math;

import java.util.List;

/**
 *
 * @author mmatula
 */
public class SyntaxErrorException extends Exception {

    private String invalidCode = "";

    public SyntaxErrorException(String msg) {
        super(msg);
    }

    public String getInvalidCode() {
        return invalidCode;
    }

    public void setInvalidCode(List<Object> code) {
        StringBuilder builder = new StringBuilder();
        for (Object object : code) {
            builder.append(String.valueOf(object));
        }
        this.invalidCode = builder.toString();
    }

    public void setInvalidCode(String code) {
        this.invalidCode = code;
    }

    public String toString() {
        return super.toString() + this.invalidCode;
    }
}
