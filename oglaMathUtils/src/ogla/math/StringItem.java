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
public class StringItem implements Code.Item {

    String string = "";

    public StringItem(String string) {
        this.string = string;
    }

    public Code.Item createCopy() {
        return new StringItem(string);
    }

    public Code.Item copy(Code.Item output) {
        StringItem stringItem = (StringItem) output;
        stringItem.string = this.string;
        return stringItem;
    }

    public boolean isCopyable() {
        return true;
    }

    public Code.Type getType() {
        return Code.Type.STRING_ITEM;
    }

    @Override
    public String toString() {
        return "StrItem: " + this.string;
    }
}
