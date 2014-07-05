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
public class BracketsItem implements Code.Item {

    Brackets brackets;
    private Code code = null;
    private Code code1 = null;
    private Code code2 = null;

    public final Code getCode() {
        return this.code;
    }

    public final void setCode(Code code) {
        this.code = code;
        this.code2 = code.createCopy();
        this.code1 = new Code();
        for (Code.Item item : code2) {
            code1.add(item);
        }
    }

    public BracketsItem(Brackets brackets, Code code) {
        this.brackets = brackets;
        this.setCode(code);
    }

    public Code.Item createCopy() {
        Code copyCode = new Code();
        for (int fa = 0; fa < this.code2.size(); fa++) {
            Code.Item item = this.code2.get(fa);
            Code.Item itemCopy = item;
            if (item.isCopyable()) {
                itemCopy = item.createCopy();
            }
            copyCode.add(itemCopy);
        }
        BracketsItem bracketsItem = new BracketsItem(brackets, copyCode);
        return bracketsItem;
    }

    public Code.Item copy(Code.Item elementObject) {
        BracketsItem bracketsItem = (BracketsItem) elementObject;
        this.code.clear();
        for (int fa = 0; fa < this.code1.size(); fa++) {
            Code.Item item = this.code1.get(fa);
            Code.Item itemCopy = bracketsItem.code.get(fa);
            itemCopy = item.copy(itemCopy);
            bracketsItem.code.add(itemCopy);
        }
        return bracketsItem;
    }

    public boolean isCopyable() {
        return true;
    }

    public Code.Type getType() {
        return Code.Type.BRACKET_ITEM;
    }

    @Override
    public String toString() {
        return this.brackets.getLeftSymbol() + code.toString() + this.brackets.getRightSymbol();
    }
}
