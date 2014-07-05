/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.math;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author mmatula
 */
public class OperatorItem implements Code.Item {

    Operator operator;

    public static class List extends ArrayList<OperatorItem> implements Code.Item {

        public Code.Item createCopy() {
            return null;
        }

        public Code.Item copy(Code.Item output) {
            return null;
        }

        public boolean isCopyable() {
            return false;
        }

        public Code.Type getType() {
            return Code.Type.OPERATOR_ITEMS;
        }

    }

    OperatorItem(Operator operator) {
        this.operator = operator;
    }

    public Code.Item createCopy() {
        return null;
    }

    public Code.Item copy(Code.Item output) {
        return null;
    }

    public boolean isCopyable() {
        return false;
    }

    public Code.Type getType() {
        return Code.Type.OPERATOR_ITEM;
    }

    
    public String toString() {
        return this.operator.toString();
    }
}
