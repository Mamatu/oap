/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.core.util;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author mmatula
 */
public class ArgumentsUtils {

    private static Map<ArgumentType, ArgumentsUtils.Checker> argumentTypesClasses = new HashMap<ArgumentType, ArgumentsUtils.Checker>();

    private static int addArgs(Writer writer, Object object, ArgumentType argumentType) {
        if (argumentTypesClasses.get(argumentType).extractBytes(object, writer)) {
            return 0;
        }
        return 2;
    }

    public static ArgumentType[] getArguments(Object arg) {
        if (arg instanceof Object[]) {
            Object[] array = (Object[]) arg;
            List<ArgumentType> args = new ArrayList<ArgumentType>();
            for (Object obj : array) {
                ArgumentType argumentType = ArgumentType.ARGUMENT_INVALID;
                for (Checker checker : argumentTypesClasses.values()) {
                    if (checker.isClassOf(obj) == true) {
                        argumentType = checker.getArgumentType();
                    }
                }
                args.add(argumentType);
            }
            ArgumentType[] argumentTypes = new ArgumentType[args.size()];
            return args.toArray(argumentTypes);
        }
        return null;
    }

    public static ByteBuffer convertObject(Object arg, ArgumentType[] argumentTypes) {
        Writer writer = new Writer();
        if (ArgumentsUtils.convertObject(writer, arg, argumentTypes) == 0) {
            return ByteBuffer.wrap(writer.getBytes());
        }
        return null;
    }

    public static int convertObject(Writer writer, Object arg, ArgumentType[] argumentTypes) {
        if (arg instanceof Object[]) {
            Object[] array = (Object[]) arg;
            arg = Arrays.asList(array);
        }
        if (arg instanceof List) {
            List objects = (List) arg;
            if (objects.size() != argumentTypes.length) {
                return 1;
            }
            for (int fa = 0; fa < objects.size(); fa++) {
                int status = ArgumentsUtils.addArgs(writer, objects.get(fa), argumentTypes[fa]);
                if (status != 0) {
                    return status;
                }
            }
            return 0;
        } else {
            if (argumentTypes.length > 1) {
                return 1;
            }
            if (argumentTypes == null && arg == null) {
                return 0;
            }
            int status = ArgumentsUtils.addArgs(writer, arg, argumentTypes[0]);
            if (status != 0) {
                return status;
            }
            return 0;
        }
    }

    private static interface Checker<T> {

        public boolean extractBytes(Object object, Writer writer);

        public byte[] getBytes(T object);

        public boolean isClassOf(Object object);

        public ArgumentType getArgumentType();
    }

    private static abstract class CheckerListImpl<T> implements Checker<T> {

        public abstract byte[] getBytes(T object);

        private Class<?> clazz = null;

        private ArgumentType argumentType = ArgumentType.ARGUMENT_INVALID;

        public CheckerListImpl(Class<?> clazz, ArgumentType argumentType) {
            this.clazz = clazz;
            this.argumentType = argumentType;
        }

        public boolean extractBytes(Object object, Writer writer) {
            if (object instanceof List) {
                List list = (List) object;
                for (Object it : list) {
                    if (clazz.isInstance(it) == false) {
                        return false;
                    }
                }
                writer.write(list.size());
                for (Object it : list) {
                    byte[] bytes = this.getBytes((T) it);
                    writer.write(bytes);
                }
                return true;
            }
            return false;
        }

        public boolean isClassOf(Object object) {
            return clazz.isInstance(object);
        }

        public ArgumentType getArgumentType() {
            return this.argumentType;
        }
    }

    private static abstract class CheckerTypeImpl<T> implements Checker<T> {

        private Class<?> clazz = null;
        private ArgumentType argumentType = ArgumentType.ARGUMENT_INVALID;

        public abstract byte[] getBytes(T object);

        public CheckerTypeImpl(Class<?> clazz, ArgumentType argumentType) {
            this.clazz = clazz;
        }

        public boolean extractBytes(Object object, Writer writer) {
            if (clazz.isInstance(object) == false) {
                return false;
            }
            byte[] bytes = this.getBytes((T) object);
            writer.write(bytes);
            return true;
        }

        public ArgumentType getArgumentType() {
            return this.argumentType;
        }

        public boolean isClassOf(Object object) {
            return clazz.isInstance(object);
        }
    }

    static {
        argumentTypesClasses.put(ArgumentType.ARGUMENT_INVALID, null);
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_ARRAY_BOOLS, new ArgumentsUtils.CheckerListImpl<Boolean>(Boolean.class, ArgumentType.ARGUMENT_TYPE_ARRAY_BOOLS) {

            @Override
            public byte[] getBytes(Boolean c) {
                byte b = 1;
                if (c == false) {
                    b = 0;
                }
                return ByteBuffer.allocate(1).order(ByteOrder.LITTLE_ENDIAN).put(b).array();
            }

            public Boolean getValue(byte[] bytes) {
                if (bytes[0] != 0) {
                    return true;
                }
                return false;
            }

            public Boolean create() {
                return null;
            }

            public Boolean getValue(ByteBuffer bytes, Boolean value) {
                byte b = bytes.get();
                if (b == 0) {
                    return new Boolean(false);
                }
                return new Boolean(true);
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_ARRAY_BYTES, new ArgumentsUtils.CheckerListImpl<Byte>(Byte.class, ArgumentType.ARGUMENT_TYPE_ARRAY_BYTES) {

            @Override
            public byte[] getBytes(Byte c) {
                return ByteBuffer.allocate(1).order(ByteOrder.LITTLE_ENDIAN).put(c.byteValue()).array();
            }

            public Byte getValue(byte[] bytes) {
                return bytes[0];
            }

            public Byte create() {
                return null;
            }

            public Byte getValue(ByteBuffer bytes, Byte value) {
                byte b = bytes.get();
                return new Byte(b);
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_ARRAY_CHARS, new ArgumentsUtils.CheckerListImpl<Character>(Character.class, ArgumentType.ARGUMENT_TYPE_ARRAY_CHARS) {

            @Override
            public byte[] getBytes(Character object) {
                byte b = (byte) object.charValue();
                return ByteBuffer.allocate(1).order(ByteOrder.LITTLE_ENDIAN).put(b).array();
            }

            public Character getValue(byte[] bytes) {
                byte b = bytes[0];
                return (char) (b);
            }

            public Character create() {
                return null;
            }

            public Character getValue(ByteBuffer bytes, Character value) {
                char c = bytes.getChar();
                return new Character(c);
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_ARRAY_DOUBLES, new ArgumentsUtils.CheckerListImpl<Double>(Double.class, ArgumentType.ARGUMENT_TYPE_ARRAY_DOUBLES) {

            @Override
            public byte[] getBytes(Double object) {
                return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putDouble(object.doubleValue()).array();
            }

            public Double getValue(byte[] bytes) {
                return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getDouble();
            }

            public Double create() {
                return null;
            }

            public Double getValue(ByteBuffer bytes, Double value) {
                double d = bytes.getDouble();
                return new Double(d);
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_ARRAY_FLOATS, new ArgumentsUtils.CheckerListImpl<Float>(Float.class, ArgumentType.ARGUMENT_TYPE_ARRAY_FLOATS) {

            @Override
            public byte[] getBytes(Float object) {
                return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(object.floatValue()).array();
            }

            public Float getValue(byte[] bytes) {
                return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getFloat();
            }

            public Float create() {
                return null;
            }

            public Float getValue(ByteBuffer bytes, Float value) {
                float f = bytes.getFloat();
                return new Float(f);
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_ARRAY_FUNCTIONS, new ArgumentsUtils.CheckerListImpl<Integer>(Integer.class, ArgumentType.ARGUMENT_TYPE_ARRAY_FUNCTIONS) {

            @Override
            public byte[] getBytes(Integer object) {
                return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(object.intValue()).array();
            }

            public Integer getValue(byte[] bytes) {
                return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
            }

            public Integer create() {
                return null;
            }

            public Integer getValue(ByteBuffer bytes, Integer value) {
                int i = bytes.getInt();
                return new Integer(i);
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_ARRAY_INTS, new ArgumentsUtils.CheckerListImpl<Integer>(Integer.class, ArgumentType.ARGUMENT_TYPE_ARRAY_INTS) {

            @Override
            public byte[] getBytes(Integer object) {
                return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(object.intValue()).array();
            }

            public Integer getValue(byte[] bytes) {
                return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
            }

            public Integer create() {
                return null;
            }

            public Integer getValue(ByteBuffer bytes, Integer value) {
                int i = bytes.getInt();
                return new Integer(i);
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_ARRAY_LONGS, new ArgumentsUtils.CheckerListImpl<Long>(Long.class, ArgumentType.ARGUMENT_TYPE_ARRAY_LONGS) {

            @Override
            public byte[] getBytes(Long object) {
                return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(object.longValue()).array();
            }

            public Long create() {
                return null;
            }

            public Long getValue(ByteBuffer bytes, Long value) {
                long l = bytes.getLong();
                return new Long(l);
            }
        });
        /*argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_ARRAY_SERIALIZED_OBJECTS, new ArgumentsUtils.CheckerListImpl<RpcSerializable>(RpcSerializable.class, ArgumentType.ARGUMENT_TYPE_ARRAY_SERIALIZED_OBJECTS) {
         @Override
         public byte[] getBytes(RpcSerializable object) {
         byte[] array = object.serialize();
         return ByteBuffer.allocate(4 + array.length).order(ByteOrder.LITTLE_ENDIAN).write(array.length).put(array).array();
         }

         public RpcSerializable create() {
         throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
         }

         public RpcSerializable getValue(ByteBuffer bytes, RpcSerializable value) {
         throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
         }

         });*/
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_ARRAY_STRINGS, new ArgumentsUtils.CheckerListImpl<String>(String.class, ArgumentType.ARGUMENT_TYPE_ARRAY_STRINGS) {

            @Override
            public byte[] getBytes(String object) {
                byte[] b = new byte[object.length()];
                for (int fa = 0; fa < object.length(); fa++) {
                    b[fa] = (byte) object.charAt(fa);
                }
                return ByteBuffer.allocate(object.length()).order(ByteOrder.LITTLE_ENDIAN).put(b).array();
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_BOOL, new ArgumentsUtils.CheckerTypeImpl<Boolean>(Boolean.class, ArgumentType.ARGUMENT_TYPE_BOOL) {

            @Override
            public byte[] getBytes(Boolean c) {
                byte b = 1;
                if (c == false) {
                    b = 0;
                }
                return ByteBuffer.allocate(1).order(ByteOrder.LITTLE_ENDIAN).put(b).array();
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_BYTE, new ArgumentsUtils.CheckerTypeImpl<Byte>(Byte.class, ArgumentType.ARGUMENT_TYPE_BYTE) {

            @Override
            public byte[] getBytes(Byte c) {
                return ByteBuffer.allocate(1).order(ByteOrder.LITTLE_ENDIAN).put(c.byteValue()).array();
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_CHAR, new ArgumentsUtils.CheckerTypeImpl<Character>(Character.class, ArgumentType.ARGUMENT_TYPE_CHAR) {

            @Override
            public byte[] getBytes(Character object) {
                byte b = (byte) object.charValue();
                return ByteBuffer.allocate(1).order(ByteOrder.LITTLE_ENDIAN).put(b).array();
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_DOUBLE, new ArgumentsUtils.CheckerTypeImpl<Double>(Double.class, ArgumentType.ARGUMENT_TYPE_DOUBLE) {

            @Override
            public byte[] getBytes(Double object) {
                return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putDouble(object.doubleValue()).array();
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_FLOAT, new ArgumentsUtils.CheckerTypeImpl<Float>(Float.class, ArgumentType.ARGUMENT_TYPE_FLOAT) {

            @Override
            public byte[] getBytes(Float object) {
                return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(object.floatValue()).array();
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_FUNCTION, new ArgumentsUtils.CheckerTypeImpl<Integer>(Integer.class, ArgumentType.ARGUMENT_TYPE_FUNCTION) {

            @Override
            public byte[] getBytes(Integer object) {
                return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(object.intValue()).array();
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_INT, new ArgumentsUtils.CheckerTypeImpl<Integer>(Integer.class, ArgumentType.ARGUMENT_TYPE_INT) {

            @Override
            public byte[] getBytes(Integer object) {
                return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(object.intValue()).array();
            }
        });
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_LONG, new ArgumentsUtils.CheckerTypeImpl<Long>(Long.class, ArgumentType.ARGUMENT_TYPE_LONG) {

            @Override
            public byte[] getBytes(Long object) {
                return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(object.longValue()).array();
            }
        });
        /*argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_SERIALIZED_OBJECT, new ArgumentsUtils.CheckerTypeImpl<RpcSerializable>(RpcSerializable.class, ArgumentType.ARGUMENT_TYPE_SERIALIZED_OBJECT) {

         @Override
         public byte[] getBytes(RpcSerializable object) {
         byte[] array = object.serialize();
         return ByteBuffer.allocate(array.length + 4).order(ByteOrder.LITTLE_ENDIAN).write(array.length).put(array).array();
         }
         });*/
        argumentTypesClasses.put(ArgumentType.ARGUMENT_TYPE_STRING, new ArgumentsUtils.CheckerTypeImpl<String>(String.class, ArgumentType.ARGUMENT_TYPE_STRING) {

            @Override
            public byte[] getBytes(String object) {
                byte[] b = new byte[object.length()];
                for (int fa = 0; fa < object.length(); fa++) {
                    b[fa] = (byte) object.charAt(fa);
                }
                return ByteBuffer.allocate(4 + object.length()).order(ByteOrder.LITTLE_ENDIAN).putInt(object.length()).put(b).array();
            }
        });
    }
}
