/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.math;

/**
 *
 * @author mmatula
 */
public class ParamItem implements Code.Item {

    private Code code = null;

    public final Code getCode() {
        return code;
    }

    ParamItem(Code code) {
        this.code = code.createCopy();
    }

    public Code.Item createCopy() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public Code.Item copy(Code.Item output) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public boolean isCopyable() {
        return false;
    }

    public Code.Type getType() {
        return Code.Type.PARAM_ITEM;
    }
}
