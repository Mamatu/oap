/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.math;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 *
 * @author mmatula
 */
public class Code implements CodeElementGiver, Iterable<Code.Item> {

    public Iterator<Item> iterator() {
        return this.code.iterator();
    }

    public enum Type {

        BRACKET_ITEM,
        MATH_STRUCTURE_ITEM,
        MATH_STRUCTURE_CREATOR_ITEM,
        MATH_STRUCTURE_CREATOR_ITEMS,
        FUNCTION_ITEM,
        FUNCTION_ITEMS,
        OPERATOR_ITEM,
        OPERATOR_ITEMS,
        VARIABLE_ITEM,
        EQUATION_ITEM,
        STRING_ITEM,
        PARAM_ITEM
    }

    public interface Item {

        public Item createCopy();

        public Item copy(Item output);

        public boolean isCopyable();

        public Type getType();
    }

    private List<Item> code = new ArrayList<Item>();

    public int size() {
        return code.size();
    }

    public Item get(int index) {
        return code.get(index);
    }

    public void add(Item object) {
        code.add(object);
    }

    public void set(int index, Item object) {
        code.set(index, object);
    }

    public void remove(int index) {
        code.remove(index);
    }

    public void clear() {
        this.code.clear();
    }

    public void insert(int index, Item obj) {
        Item e = obj;
        for (int fa = index; fa < code.size(); fa++) {
            e = code.set(fa, e);
        }
        code.add(e);
    }

    public Code createCopy() {
        Code copy = new Code();
        for (int fa = 0; fa < code.size(); fa++) {
            Code.Item item = code.get(fa);
            Code.Item itemCopy = item;
            if (item.isCopyable()) {
                itemCopy = item.createCopy();
            }
            copy.add(itemCopy);
        }
        return copy;
    }

    private Map<Code.Item, List<Code.Item>> copies = new IdentityHashMap<Code.Item, List<Code.Item>>();

    public Code.Item getCopy(Code.Item codeElement) {
        Code.Item copy = this.copies.get(codeElement).get(0);
        return copy;
    }

    @Override
    public String toString() {
        return code.toString();
    }
}
