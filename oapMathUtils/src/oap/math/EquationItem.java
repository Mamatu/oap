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
public class EquationItem implements Code.Item {

    String string = "";

    public EquationItem(String string) {
        this.string = string;
    }

    public Code.Item createCopy() {
        return new EquationItem(string);
    }

    public Code.Item copy(Code.Item output) {
        EquationItem stringItem = (EquationItem) output;
        stringItem.string = this.string;
        return stringItem;
    }

    public boolean isCopyable() {
        return true;
    }

    public Code.Type getType() {
        return Code.Type.EQUATION_ITEM;
    }

    @Override
    public String toString() {
        return "EqItem: " + this.string;
    }

}
