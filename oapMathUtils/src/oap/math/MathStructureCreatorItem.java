/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.math;

import java.util.ArrayList;

/**
 *
 * @author mmatula
 */
public class MathStructureCreatorItem implements Code.Item {

    public static class List extends ArrayList<MathStructureCreatorItem> implements Code.Item {

        Code code;

        public List(Code code) {
            this.code = code;
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
            return Code.Type.MATH_STRUCTURE_CREATOR_ITEMS;
        }
    }

    MathStructure.Creator creator;

    public MathStructureCreatorItem(MathStructure.Creator creator) {
        this.creator = creator;
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
        return Code.Type.MATH_STRUCTURE_CREATOR_ITEM;
    }

    public String toString() {
        return this.creator.toString();
    }
}
