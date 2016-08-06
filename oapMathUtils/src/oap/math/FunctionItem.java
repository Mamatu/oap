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
public class FunctionItem implements Code.Item {

    Function function;

    public static class List extends ArrayList<FunctionItem> implements Code.Item {

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
            return Code.Type.FUNCTION_ITEMS;
        }
    }

    public FunctionItem(Function function) {
        this.function = function;
    }

    public Code.Item createCopy() {
        return null;
    }

    public Code.Item copy(Code.Item elementObject) {
        return null;
    }

    public boolean isCopyable() {
        return true;
    }

    public Code.Type getType() {
        return Code.Type.FUNCTION_ITEM;
    }

    public String toString() {
        return this.function.toString();
    }
}
