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
public class MathStructureItem implements Code.Item {

    MathStructure mathStructure;

    public MathStructureItem(MathStructure mathStructure) {
        this.mathStructure = mathStructure;
    }

    public Code.Item createCopy() {
        MathStructure mathStructure = this.mathStructure.createCopy();
        MathStructureItem mathStructureItem = new MathStructureItem(mathStructure);
        return mathStructureItem;
    }

    public Code.Item copy(Code.Item elementObject) {
        MathStructureItem mathStructureItem = (MathStructureItem) elementObject;
        mathStructureItem.mathStructure = this.mathStructure.copy(mathStructureItem.mathStructure);
        return mathStructureItem;
    }

    public boolean isCopyable() {
        return true;
    }

    public Code.Type getType() {
        return Code.Type.MATH_STRUCTURE_ITEM;
    }

    public String toString() {
        return this.mathStructure.toString();
    }

}
