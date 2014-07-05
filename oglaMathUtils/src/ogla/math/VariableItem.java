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
public class VariableItem implements Code.Item {

    Variable variable;

    VariableItem(Variable variable) {
        this.variable = variable;
    }

    public Code.Item createCopy() {
        Variable variable = new Variable(this.variable.name);
        VariableItem variableItem = new VariableItem(this.variable);
        return variableItem;
    }

    public Code.Item copy(Code.Item elementObject) {
        VariableItem variableItem = (VariableItem) elementObject;
        variableItem.variable.name = this.variable.name;
        return variableItem;
    }

    public boolean isCopyable() {
        return true;
    }

    public Code.Type getType() {
        return Code.Type.VARIABLE_ITEM;
    }

    public String toString() {
        return this.variable.name;
    }
}
