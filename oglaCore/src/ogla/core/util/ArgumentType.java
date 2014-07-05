/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.core.util;

/**
 *
 * @author mmatula
 */
public enum ArgumentType {

    ARGUMENT_INVALID(-1),
    ARGUMENT_TYPE_INT,
    ARGUMENT_TYPE_LONG,
    ARGUMENT_TYPE_FLOAT,
    ARGUMENT_TYPE_DOUBLE,
    ARGUMENT_TYPE_CHAR,
    ARGUMENT_TYPE_STRING,
    ARGUMENT_TYPE_BOOL,
    ARGUMENT_TYPE_BYTE,
    ARGUMENT_TYPE_SERIALIZED_OBJECT,
    ARGUMENT_TYPE_FUNCTION,
    ARGUMENT_TYPE_ARRAY_INTS,
    ARGUMENT_TYPE_ARRAY_LONGS,
    ARGUMENT_TYPE_ARRAY_FLOATS,
    ARGUMENT_TYPE_ARRAY_DOUBLES,
    ARGUMENT_TYPE_ARRAY_CHARS,
    ARGUMENT_TYPE_ARRAY_STRINGS,
    ARGUMENT_TYPE_ARRAY_BOOLS,
    ARGUMENT_TYPE_ARRAY_BYTES,
    ARGUMENT_TYPE_ARRAY_SERIALIZED_OBJECTS,
    ARGUMENT_TYPE_ARRAY_FUNCTIONS;

    private final int type;

    private static int nextValue;

    public static final ArgumentType fromInteger(int id) {
        if (id == -1) {
            return ARGUMENT_INVALID;
        } else if (id >= 0) {
            ArgumentType[] args = ArgumentType.values();
            if (id + 1 < args.length) {
                return args[id + 1];
            } else {
                return ARGUMENT_INVALID;
            }
        }
        return ARGUMENT_INVALID;
    }

    public static final int toInteger(ArgumentType at) {
        return at.ordinal() - 1;
    }

    public static final String toString(ArgumentType at) {
        if (at == ArgumentType.ARGUMENT_INVALID) {
            return "INVALID";
        } else if (at == ArgumentType.ARGUMENT_TYPE_ARRAY_BOOLS) {
            return "INVALID";
        } else if (at == ArgumentType.ARGUMENT_TYPE_ARRAY_BYTES) {
            return "ARRAY_BYTES";
        } else if (at == ArgumentType.ARGUMENT_TYPE_ARRAY_CHARS) {
            return "ARRAY_CHARS";
        } else if (at == ArgumentType.ARGUMENT_TYPE_ARRAY_DOUBLES) {
            return "ARRAY_DOUBLES";
        } else if (at == ArgumentType.ARGUMENT_TYPE_ARRAY_FLOATS) {
            return "ARRAY_FLOATS";
        } else if (at == ArgumentType.ARGUMENT_TYPE_ARRAY_FUNCTIONS) {
            return "ARRAY_FUNCTIONS";
        } else if (at == ArgumentType.ARGUMENT_TYPE_ARRAY_INTS) {
            return "ARRAY_INTS";
        } else if (at == ArgumentType.ARGUMENT_TYPE_ARRAY_LONGS) {
            return "ARRAY_LONGS";
        } else if (at == ArgumentType.ARGUMENT_TYPE_ARRAY_SERIALIZED_OBJECTS) {
            return "ARRAY_SERIALIZED_OBJECTS";
        } else if (at == ArgumentType.ARGUMENT_TYPE_ARRAY_STRINGS) {
            return "ARRAY_STRINGS";
        } else if (at == ArgumentType.ARGUMENT_TYPE_BOOL) {
            return "BOOL";
        } else if (at == ArgumentType.ARGUMENT_TYPE_BYTE) {
            return "BYTE";
        } else if (at == ArgumentType.ARGUMENT_TYPE_CHAR) {
            return "CHAR";
        } else if (at == ArgumentType.ARGUMENT_TYPE_DOUBLE) {
            return "DOUBLE";
        } else if (at == ArgumentType.ARGUMENT_TYPE_FLOAT) {
            return "FLOAT";
        } else if (at == ArgumentType.ARGUMENT_TYPE_FUNCTION) {
            return "FUNCTION";
        } else if (at == ArgumentType.ARGUMENT_TYPE_INT) {
            return "INT";
        } else if (at == ArgumentType.ARGUMENT_TYPE_LONG) {
            return "LONG";
        } else if (at == ArgumentType.ARGUMENT_TYPE_SERIALIZED_OBJECT) {
            return "SERIALIZED_OBJECT";
        } else if (at == ArgumentType.ARGUMENT_TYPE_STRING) {
            return "STRING";
        } else {
            return "VOID";
        }
    }

    ArgumentType(int type) {
        this.type = type;
        Counter.nextValue = type + 1;
    }

    ArgumentType() {
        this(Counter.nextValue);
    }

    private static class Counter {
        private static int nextValue = 0;
    }
}
