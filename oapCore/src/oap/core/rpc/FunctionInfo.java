/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.core.rpc;

import ogla.core.util.ArgumentType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class FunctionInfo {

    private Object connection = null;

    public FunctionInfo(Object connection) {
        this.connection = connection;
    }

    public FunctionInfo(String name, ArgumentType[] inargs, ArgumentType[] outargs, Object connection) {
        this.setName(name);
        this.setInputArgs(inargs);
        this.setOutputArgs(inargs);
        this.connection = connection;
    }

    private void fillNames(RpcObject object) {
        if (object != null) {
            this.fillNames(object.getRoot());
            this.names.add(object.getName());
        }
    }

    protected String[] getNamesRef() {
        if (allnames == null) {
            allnames = new String[this.names.size() + 1];
            allnames = this.names.toArray(allnames);
            allnames[allnames.length - 1] = name;
        }
        return allnames;
    }

    protected ArgumentType[] getOutputArgsRef() {
        return this.outargs;
    }

    protected ArgumentType[] getInputArgsRef() {
        return this.inargs;
    }

    public FunctionInfo(RpcObject object, String name, ArgumentType[] inargs, ArgumentType[] outargs, Object connection) {
        this.setName(name);
        this.setInputArgs(inargs);
        this.setOutputArgs(inargs);
        this.setObjects(object);
        this.connection = connection;
    }

    public FunctionInfo(String[] objectNames, String name, ArgumentType[] inargs, ArgumentType[] outargs, Object connection) {
        this.setName(name);
        this.setInputArgs(inargs);
        this.setOutputArgs(inargs);
        this.setObjectsNames(objectNames);
        this.connection = connection;
    }

    public boolean equalsArgumentType(ArgumentType[] argumentTypes) {
        if (this.inargs != null && argumentTypes.length == inargs.length) {
            for (int fa = 0; fa < argumentTypes.length; fa++) {
                if (argumentTypes[fa] != inargs[fa]) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    private ArgumentType[] inargs = null;
    private ArgumentType[] outargs = null;
    private String name = "";
    private List<String> names = new ArrayList<String>();
    private String[] allnames = null;

    protected final void setName(String name) {
        this.name = name;
    }

    protected final void setObjects(RpcObject object) {
        this.fillNames(object);
    }

    protected final void setNames(String[] names) {
        if (names != null && names.length > 0) {
            if (names.length > 1) {
                String[] names1 = new String[names.length - 1];
                System.arraycopy(names, 0, names1, 0, names1.length);
                this.names.addAll(Arrays.asList(names1));
            }
            this.setName(names[names.length - 1]);
        }
    }

    protected final void setObjectsNames(String[] names) {
        if (names != null && names.length > 0) {
            this.names.addAll(Arrays.asList(names));
        } else {
            this.names = null;
        }
    }

    protected final void setInputArgs(ArgumentType[] args) {
        if (args != null) {
            this.inargs = new ArgumentType[args.length];
            System.arraycopy(args, 0, this.inargs, 0, args.length);
        }
    }

    protected final void setOutputArgs(ArgumentType[] args) {
        if (args != null) {
            this.outargs = new ArgumentType[args.length];
            System.arraycopy(args, 0, this.outargs, 0, args.length);
        }
    }

    public final int getObjectsNamesCount() {
        if (names == null) {
            return 0;
        }
        return names.size();
    }

    public final String[] getObjectsNames(String[] names) {
        if (this.names == null) {
            return names;
        }
        System.arraycopy(this.names.toArray(), 0, names, 0, names.length < this.names.size() ? names.length : this.names.size());
        return names;
    }

    public final String getName() {
        return this.name;
    }

    public final int getInputArgumentsCount() {
        if (this.inargs != null) {
            return this.inargs.length;
        }
        return 0;
    }

    public final int getOutputArgumentsCount() {
        if (this.outargs != null) {
            return this.outargs.length;
        }
        return 0;
    }

    public final ArgumentType[] getInputArguments(ArgumentType[] array) {
        if (array != null && array.length > 0) {
            System.arraycopy(this.inargs, 0, array, 0, array.length <= this.inargs.length ? array.length : this.inargs.length);
        }
        return array;
    }

    public final ArgumentType[] getOutputArguments(ArgumentType[] array) {
        if (array != null && array.length > 0) {
            System.arraycopy(this.outargs, 0, array, 0, array.length <= this.outargs.length ? array.length : this.outargs.length);
        }
        return array;
    }

    public Object getConnection() {
        return this.connection;
    }
}
